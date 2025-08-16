import asyncio
import os
import tempfile
import shutil
import logging
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyrogram import Client, filters as pyrogram_filters
from pyrogram.types import Message
import psutil

# Configure logging
logging.basicConfig(
    format='%(levelname)s: %(message)s',
    level=logging.WARNING)
logger = logging.getLogger(__name__)


class WatermarkBot:

    def __init__(self, api_id: int, api_hash: str, bot_token: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.bot_token = bot_token
        self.user_sessions = {}
        self.persistent_data = {}
        self.admin_users = [5838583307]  # Admin user IDs

        # Create directories for persistent storage including normalized intros cache
        for dir_name in [
                'persistent_intros', 'persistent_watermarks',
                'persistent_thumbnails', 'bulk_queue', 'persistent_captions',
                'normalized_intros_cache'
        ]:
            try:
                os.makedirs(dir_name, mode=0o755, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {dir_name}: {e}")
                raise

        # Initialize Pyrogram client with fresh session
        self.app = Client("watermark_bot_new",
                          api_id=api_id,
                          api_hash=api_hash,
                          bot_token=bot_token,
                          workdir=".")

    def get_system_usage(self):
        """Get CPU and RAM usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            return f"💻 CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%"
        except:
            return "💻 System info unavailable"

    async def create_progress_message(self,
                                      message: Message,
                                      initial_text: str,
                                      video_num: int = None,
                                      total_videos: int = None,
                                      filename: str = None):
        """Create progress message with enhanced tracking and user info"""
        user_id = message.from_user.id

        # Store user information for progress display
        user_info = {
            'name': getattr(message.from_user, 'first_name', f'User{user_id}'),
            'id': user_id
        }

        # Get task name from filename, video, or caption - prioritize actual filename
        task_name = "Processing.mp4"
        if filename:
            task_name = filename
        elif hasattr(message, 'video') and message.video:
            task_name = getattr(message.video, 'file_name', message.caption or "Video.mp4")
        elif hasattr(message, 'caption') and message.caption:
            task_name = message.caption[:30] + ".mp4" if len(message.caption) > 30 else message.caption + ".mp4"

        # Truncate filename if too long for display
        if len(task_name) > 35:
            task_name = task_name[:32] + "..."

        # Add video counter if bulk processing
        if video_num and total_videos:
            counter_text = f"[{video_num}/{total_videos}] "
            initial_text = counter_text + initial_text

        progress_message = await message.reply_text(initial_text)

        # Store in session for enhanced tracking
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}

        self.user_sessions[user_id].update({
            'progress_message': progress_message,
            'video_counter': f"[{video_num}/{total_videos}] " if video_num and total_videos else "",
            'last_update_time': 0,
            'last_percent': -1,
            'update_count': 0,
            'user_info': user_info,
            'current_task': task_name,
            'download_started': False,
            'bytes_downloaded': 0,
            'download_speed_samples': [],
            'last_speed_update': 0
        })

        return progress_message

    async def update_progress_smooth(self,
                                     progress_message,
                                     stage: str,
                                     percent: float,
                                     details: dict = None,
                                     video_counter: str = "",
                                     force_update: bool = False):
        """Enhanced progress update with fixed percentage calculation and better UI"""
        current_time = time.time()
        user_id = None

        # Find user_id from sessions
        for uid, session in self.user_sessions.items():
            if session.get('progress_message') == progress_message:
                user_id = uid
                break

        if not user_id:
            return

        session = self.user_sessions.get(user_id, {})
        last_update_time = session.get('last_update_time', 0)
        last_percent = session.get('last_percent', -1)
        update_count = session.get('update_count', 0)

        # Ensure percent is valid and within bounds
        if percent is None or percent < 0:
            percent = 0
        elif percent > 100:
            percent = 100

        # Update every 2 seconds for better responsiveness
        time_since_last = current_time - last_update_time
        percent_change = abs(percent - last_percent)

        should_update = (force_update or percent >= 100
                         or percent_change >= 1.0 or time_since_last >= 2.0
                         or update_count == 0)

        if not should_update:
            return

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Get user info and task details
                user_info = session.get('user_info', {})
                user_name = user_info.get('name', f'User{user_id}')
                task_name = session.get('current_task', 'Processing')

                # Enhanced progress bar with better visual design
                bar_length = 12
                filled = int(bar_length * percent / 100)
                bar_chars = ["█"] * filled + ["░"] * (bar_length - filled)
                bar = "".join(bar_chars)

                # Stage icons and descriptions with more detail
                stage_mapping = {
                    "downloading": ("Downloading", "⬇️"),
                    "pipeline": ("Processing", "🔄"),
                    "watermarking": ("Watermarking", "🎨"),
                    "processing": ("Processing", "⚙️"),
                    "encoding": ("Encoding", "🎬"),
                    "finalizing": ("Finalizing", "🔧"),
                    "uploaded": ("Uploading", "📤"),
                    "uploading": ("Uploading", "📤"),
                    "completed": ("Completed", "✅"),
                    "rendering": ("Rendering", "🎭"),
                    "concatenating": ("Merging", "🔗"),
                    "normalizing": ("Normalizing", "📐"),
                    "queued": ("Queued", "⏳"),
                    "initializing": ("Initializing", "🚀")
                }

                stage_text, stage_icon = stage_mapping.get(stage, ("Processing", "🔄"))

                # Build enhanced progress display
                system_usage = self.get_system_usage()
                progress_text = f"**📂 Task1: {task_name}**\n"
                progress_text += f"├─ `{bar}` **{percent:.1f}%**\n"
                progress_text += f"├─ **Status:** {stage_text} {stage_icon}\n"
                progress_text += f"├─ **System:** {system_usage}\n"

                # Add detailed processing information
                if details:
                    # Enhanced file transfer tracking - prioritize downloaded amount display
                    if 'current_mb' in details:
                        current_mb = max(0, float(details.get('current_mb', 0)))

                        if current_mb > 0:
                            if 'total_mb' in details and details['total_mb'] and float(details['total_mb']) > 0:
                                total_mb = float(details['total_mb'])
                                # Only show total if it's reasonable (greater than current)
                                if total_mb >= current_mb:
                                    progress_text += f"├─ **Progress:** {current_mb:.2f}MB / {total_mb:.2f}MB\n"
                                else:
                                    progress_text += f"├─ **Downloaded:** {current_mb:.2f}MB\n"
                            else:
                                progress_text += f"├─ **Downloaded:** {current_mb:.2f}MB\n"

                    # Speed information with better formatting
                    if 'speed_mbps' in details and details['speed_mbps']:
                        speed = float(details['speed_mbps'])
                        if speed > 0:
                            if speed >= 1:
                                progress_text += f"├─ **Speed:** {speed:.2f}MB/s ⚡\n"
                            elif speed >= 0.1:
                                progress_text += f"├─ **Speed:** {speed:.3f}MB/s ⚡\n"
                            else:
                                progress_text += f"├─ **Speed:** {speed*1024:.1f}KB/s ⚡\n"

                    # Frame processing details
                    if 'frame_progress' in details and details['frame_progress']:
                        progress_text += f"├─ **Frames:** {details['frame_progress']}\n"

                    # Processing FPS
                    if 'processing_fps' in details and details['processing_fps'] and float(details['processing_fps']) > 0:
                        fps_val = float(details['processing_fps'])
                        progress_text += f"├─ **Processing:** {fps_val:.1f} fps 🎬\n"

                    # ETA information
                    if 'eta' in details and details['eta'] and str(details['eta']).strip():
                        progress_text += f"├─ **ETA:** {details['eta']} ⏱️\n"
                    else:
                        progress_text += f"├─ **ETA:** Calculating... ⏱️\n"

                    # Elapsed time
                    if 'elapsed' in details and details['elapsed'] and str(details['elapsed']).strip():
                        progress_text += f"└─ **Elapsed:** {details['elapsed']} ⏰\n"
                    else:
                        # Remove the last ├─ and replace with └─ if no elapsed time
                        progress_text = progress_text.rstrip('\n')
                        if progress_text.endswith('├─'):
                            progress_text = progress_text[:-2] + '└─'
                        progress_text += "\n"
                else:
                    # Close the tree structure
                    progress_text = progress_text.rstrip('\n')
                    if progress_text.endswith('├─'):
                        progress_text = progress_text[:-2] + '└─'
                    progress_text += "\n"

                await progress_message.edit_text(progress_text)

                # Update session tracking
                session['last_update_time'] = current_time
                session['last_percent'] = percent
                session['update_count'] = update_count + 1

                break

            except Exception as e:
                retry_count += 1
                error_str = str(e)

                if "FLOOD_WAIT" in error_str:
                    try:
                        wait_time = int(error_str.split("FLOOD_WAIT_")[1].split()[0])
                        await asyncio.sleep(min(wait_time, 5))
                    except:
                        await asyncio.sleep(2)
                elif "MESSAGE_NOT_MODIFIED" in error_str:
                    session['last_update_time'] = current_time
                    session['last_percent'] = percent
                    break
                elif "Request timed out" in error_str or "NetworkError" in error_str:
                    await asyncio.sleep(0.5)
                else:
                    if retry_count >= max_retries:
                        logger.warning(f"Progress update failed after {max_retries} attempts: {e}")
                        break
                    await asyncio.sleep(0.2)

    async def download_with_progress(self,
                                     message: Message,
                                     file_id: str,
                                     file_path: str,
                                     file_type: str,
                                     video_num: int = None,
                                     total_videos: int = None,
                                     filename: str = None):
        """Download file with enhanced progress tracking - no file size limits"""
        user_id = message.from_user.id

        # Check if process was stopped for this specific user
        session = self.user_sessions.get(user_id, {})
        if session.get('stopped'):
            raise Exception(f"Process was stopped by user {user_id}")

        # Get file size if available (no size limits enforced)
        file_size = 0
        try:
            # Try to get file size from message object first
            if hasattr(message, 'video') and message.video and hasattr(message.video, 'file_size') and message.video.file_size:
                file_size = message.video.file_size
            elif hasattr(message, 'photo') and message.photo and hasattr(message.photo, 'file_size') and message.photo.file_size:
                file_size = message.photo.file_size
            elif hasattr(message, 'document') and message.document and hasattr(message.document, 'file_size') and message.document.file_size:
                file_size = message.document.file_size

            # If still no file size, try Telegram API
            if file_size == 0 or file_size is None:
                try:
                    file_info = await self.app.get_file(file_id)
                    if hasattr(file_info, 'file_size') and file_info.file_size:
                        file_size = file_info.file_size

                except Exception as e:
                    logger.warning(f"Could not get file size from API: {e}")


        except Exception as e:
            logger.warning(f"Error getting file size: {e}")

        # Use filename for task display if provided
        display_filename = filename or (
            getattr(message.video, 'file_name', None) if hasattr(message, 'video') and message.video else None
        ) or message.caption or f"{file_type}_file"

        # Create progress message with filename display
        if "video" in file_type.lower():
            progress_message = await self.create_progress_message(
                message, f"🔄 **Processing video pipeline...**", video_num, total_videos, display_filename)
        else:
            progress_message = await self.create_progress_message(
                message, f"⬇️ **Downloading {file_type}...**", video_num, total_videos, display_filename)

        video_counter = self.user_sessions[user_id].get('video_counter', "")

        start_time = time.time()
        speed_samples = []
        last_current = 0
        last_callback_time = start_time
        session['download_started'] = True

        async def enhanced_progress_callback(current, total):
            nonlocal speed_samples, last_current, last_callback_time
            current_time = time.time()

            # Check if stopped
            if self.user_sessions.get(user_id, {}).get('stopped'):
                raise Exception("Process was stopped by user")

            # Update session with latest download progress
            session['bytes_downloaded'] = current

            # Smart total size handling - prioritize actual data over estimates
            working_total = total

            # Use detected file size if total is invalid or zero
            if working_total <= 0 or working_total is None:
                if file_size > 0:
                    working_total = file_size
                else:
                    working_total = 0  # Will show download amount only

            # Handle case where current exceeds reported total (adjust total upward)
            if current > working_total and working_total > 0:
                working_total = current + (current * 0.1)  # Add 10% buffer
                logger.info(f"Adjusted total size upward: {working_total} bytes")

            # Calculate download metrics
            current_mb = current / (1024 * 1024)
            total_mb = working_total / (1024 * 1024) if working_total > 0 else 0
            elapsed = current_time - start_time

            # Enhanced speed calculation with better smoothing
            time_diff = current_time - last_callback_time
            if time_diff >= 0.5 and current > last_current:  # At least 500ms for stable measurement
                bytes_diff = current - last_current
                instant_speed = bytes_diff / time_diff

                # Add to rolling average (keep last 8 samples)
                speed_samples.append(instant_speed)
                if len(speed_samples) > 8:
                    speed_samples.pop(0)

                # Use weighted average (recent samples have higher weight)
                if len(speed_samples) >= 3:
                    weights = [i + 1 for i in range(len(speed_samples))]
                    weighted_sum = sum(speed * weight for speed, weight in zip(speed_samples, weights))
                    weight_total = sum(weights)
                    avg_speed = weighted_sum / weight_total
                else:
                    avg_speed = sum(speed_samples) / len(speed_samples)

                speed_mbps = avg_speed / (1024 * 1024)
            elif elapsed > 0 and current > 0:
                # Fallback to overall average speed
                avg_speed = current / elapsed
                speed_mbps = avg_speed / (1024 * 1024)
            else:
                speed_mbps = 0

            # Calculate percentage - prioritize actual progress over estimates
            if working_total > 0 and current <= working_total:
                percent = (current / working_total) * 100
            elif current > 0:
                # Show progress without percentage if no reliable total
                percent = min(99, (current_mb / 10) * 10)  # Rough progress based on downloaded amount
            else:
                percent = 0

            # ETA calculation
            if speed_mbps > 0 and working_total > 0 and current < working_total:
                remaining_bytes = working_total - current
                eta_seconds = remaining_bytes / (speed_mbps * 1024 * 1024)

                if eta_seconds > 3600:
                    eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"
                elif eta_seconds > 60:
                    eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                else:
                    eta_str = f"{int(eta_seconds)}s"
            else:
                eta_str = "calculating..." if current < working_total else "finishing..."

            # Format elapsed time
            if elapsed > 3600:
                elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
            elif elapsed > 60:
                elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
            else:
                elapsed_str = f"{int(elapsed)}s"

            # Build progress details - always show downloaded amount
            details = {
                'current_mb': current_mb,
                'elapsed': elapsed_str
            }

            # Add total size only if we have a reliable value
            if working_total > 0 and total_mb >= current_mb:
                details['total_mb'] = total_mb
                details['eta'] = eta_str

            # Add speed only if meaningful
            if speed_mbps > 0:
                details['speed_mbps'] = speed_mbps

            # Update progress every 2 seconds or on significant changes
            force_update = (current >= working_total) if working_total > 0 else False
            stage_name = "pipeline" if "video" in file_type.lower() else "downloading"

            if (current_time - last_callback_time >= 2.0) or force_update or (current == 0):
                await self.update_progress_smooth(
                    progress_message, stage_name, percent, details, video_counter, force_update
                )
                last_callback_time = current_time

            last_current = current

        try:
            await self.app.download_media(file_id, file_name=file_path, progress=enhanced_progress_callback)

            # Final completion update
            final_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            final_mb = final_size / (1024 * 1024)

            await self.update_progress_smooth(
                progress_message, "completed", 100,
                {
                    'current_mb': final_mb,
                    'status': f'{file_type.title()} download completed',
                    'elapsed': f"{int(time.time() - start_time)}s"
                },
                video_counter, True
            )

        except Exception as e:
            if "stopped by user" in str(e):
                raise
            else:
                logger.error(f"Download error: {e}")
                raise Exception(f"Download failed: {str(e)}")

        return progress_message

    async def get_video_info(self, video_path: str):
        """Get comprehensive video information using ffprobe"""
        import subprocess
        import json

        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            info = json.loads(result.stdout)

            video_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if video_stream:
                # Parse frame rate
                fps_str = video_stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) > 0 else 30
                else:
                    fps = float(fps_str)

                # Get duration from multiple sources
                duration = 0
                if video_stream.get('duration'):
                    duration = float(video_stream.get('duration'))
                elif info.get('format', {}).get('duration'):
                    duration = float(info.get('format', {}).get('duration'))
                elif video_stream.get('tags', {}).get('DURATION'):
                    duration_str = video_stream.get('tags', {}).get('DURATION')
                    try:
                        parts = duration_str.split(':')
                        if len(parts) >= 3:
                            hours = float(parts[0])
                            minutes = float(parts[1])
                            seconds = float(parts[2])
                            duration = hours * 3600 + minutes * 60 + seconds
                    except:
                        pass

                return {
                    'width':
                    int(video_stream.get('width', 1280)),
                    'height':
                    int(video_stream.get('height', 720)),
                    'fps':
                    fps,
                    'duration':
                    duration,
                    'codec':
                    video_stream.get('codec_name', 'h264'),
                    'bitrate':
                    int(video_stream.get('bit_rate', 0))
                    if video_stream.get('bit_rate') else 0
                }
        except Exception as e:
            logger.warning(f"Could not get video info for {video_path}: {e}")

        return {
            'width': 1280,
            'height': 720,
            'fps': 30,
            'duration': 0,
            'codec': 'h264',
            'bitrate': 0
        }

    async def normalize_video(self, input_path: str, output_path: str,
                              target_width: int, target_height: int,
                              target_fps: float):
        """Normalize video to target specifications with metadata preservation"""
        import subprocess

        try:
            # Get input video info
            input_info = await self.get_video_info(input_path)

            # Build normalization command with metadata preservation
            cmd = [
                'ffmpeg',
                '-i',
                input_path,
                '-vf',
                f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,fps={target_fps}',
                '-c:v',
                'libx264',
                '-preset',
                'medium',
                '-crf',
                '20',
                '-c:a',
                'aac',
                '-ar',
                '44100',
                '-ac',
                '2',
                '-b:a',
                '128k',
                '-maxrate',
                '3M',
                '-bufsize',
                '6M',
                '-map_metadata',
                '0',  # Preserve metadata
                '-movflags',
                '+faststart',
                '-y',
                output_path
            ]

            result = subprocess.run(cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True)
            logger.info(
                f"Successfully normalized video: {input_path} -> {output_path}"
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Video normalization failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in video normalization: {e}")
            return False

    async def normalize_video_with_progress(self,
                                            input_path: str,
                                            output_path: str,
                                            target_width: int,
                                            target_height: int,
                                            target_fps: float,
                                            progress_callback=None):
        """Normalize video with detailed progress tracking"""
        import subprocess

        try:
            # Get input video info
            input_info = await self.get_video_info(input_path)
            total_frames = int(target_fps * input_info.get('duration', 0))

            # Build normalization command with metadata preservation
            cmd = [
                'ffmpeg',
                '-i',
                input_path,
                '-vf',
                f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,fps={target_fps}',
                '-c:v',
                'libx264',
                '-preset',
                'medium',
                '-crf',
                '20',
                '-c:a',
                'aac',
                '-ar',
                '44100',
                '-ac',
                '2',
                '-b:a',
                '128k',
                '-maxrate',
                '3M',
                '-bufsize',
                '6M',
                '-map_metadata',
                '0',  # Preserve metadata
                '-movflags',
                '+faststart',
                '-y',
                output_path
            ]

            if progress_callback:
                process = subprocess.Popen(cmd,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)

                frame_count = 0
                last_frame_update = time.time()
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        if 'frame=' in output:
                            try:
                                current_time = time.time()
                                frame_match = output.split(
                                    'frame=')[1].split()[0]
                                frame_count = int(frame_match)
                                # Update progress every 10 seconds for flood protection
                                if current_time - last_frame_update >= 10.0:
                                    if total_frames > 0:
                                        percent = 10 + (
                                            frame_count /
                                            total_frames
                                        ) * 50  # 10-60% for normalization
                                        await progress_callback(
                                            percent,
                                            100,
                                            f"Normalizing intro: frame {frame_count}/{total_frames}"
                                        )
                                    else:
                                        await progress_callback(
                                            30,
                                            100,
                                            f"Normalizing intro: frame {frame_count}"
                                        )
                                    last_frame_update = current_time
                            except:
                                pass

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, cmd)
            else:
                result = subprocess.run(cmd,
                                        check=True,
                                        capture_output=True,
                                        text=True)

            logger.info(
                f"Successfully normalized video with progress: {input_path} -> {output_path}"
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Video normalization with progress failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error in video normalization with progress: {e}")
            return False

    async def generate_video_thumbnail(self, video_path: str,
                                       thumbnail_path: str):
        """Generate thumbnail from video at 10% duration point"""
        import subprocess

        try:
            # Get video duration
            video_info = await self.get_video_info(video_path)
            duration = video_info.get('duration', 0)

            # Extract frame at 10% of video duration (or 3 seconds if duration unknown)
            seek_time = max(3, duration * 0.1) if duration > 0 else 3

            cmd = [
                'ffmpeg',
                '-i',
                video_path,
                '-ss',
                str(seek_time),
                '-vf',
                'scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2',
                '-vframes',
                '1',
                '-q:v',
                '2',  # High quality thumbnail
                '-y',
                thumbnail_path
            ]

            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            return os.path.exists(thumbnail_path)

        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return False

    async def add_combined_watermarks(self,
                                      input_path: str,
                                      output_path: str,
                                      watermark_text: str = None,
                                      watermark_png_path: str = None,
                                      png_location: str = "topright",
                                      progress_callback=None):
        """Add combined watermarks with speed optimization and progress tracking"""
        import subprocess

        try:
            # Get video info
            video_info = await self.get_video_info(input_path)
            width = video_info['width']
            height = video_info['height']
            fps = video_info['fps']
            duration = video_info.get('duration', 0)

            if fps <= 0 or fps > 120:
                fps = 30

            # Calculate total frames for progress tracking
            total_frames = int(fps * duration) if duration > 0 else 0

            # Build filter complex
            filter_parts = []
            overlay_inputs = "[0:v]"

            # Add text watermark if provided and not skipped
            if watermark_text and watermark_text.strip():
                # Clean and escape the text for FFmpeg, but preserve actual line breaks
                clean_text = watermark_text.replace("'", "\\'").replace(":", "\\:")
                # Remove any characters that might break FFmpeg filters but keep newlines
                clean_text = ''.join(
                    c for c in clean_text
                    if ord(c) < 127 and (c.isprintable() or c in ['\n', '\r', ' ']))

                # Split by actual line breaks first (user's intended lines)
                user_lines = clean_text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

                # Process each user line for length and create final lines
                lines = []
                max_chars_per_line = max(25, width // 12)  # Slightly longer lines for better readability

                for user_line in user_lines:
                    user_line = user_line.strip()
                    if not user_line and len(lines) == 0:  # Skip empty lines only at the beginning
                        continue
                    elif not user_line:  # Preserve empty lines in the middle/end as spacing
                        lines.append(" ")  # Use space instead of empty to maintain line structure
                        continue

                    # If line is short enough, use as-is
                    if len(user_line) <= max_chars_per_line:
                        lines.append(user_line)
                    else:
                        # Split long lines by words
                        words = user_line.split()
                        current_line = ""

                        for word in words:
                            test_line = current_line + " " + word if current_line else word
                            if len(test_line) <= max_chars_per_line:
                                current_line = test_line
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word

                        if current_line:
                            lines.append(current_line)

                # Limit to 5 lines maximum for better display (increased from 4)
                if len(lines) > 5:
                    lines = lines[:4]
                    lines.append("...")

                # Join lines with proper newline character for FFmpeg
                # Use \\n for proper line breaks in FFmpeg drawtext
                final_text = "\\n".join(lines)

                font_size = max(18, min(width // 40, height //
                                        30))  # Slightly smaller for multi-line
                margin = 15
                line_spacing = int(font_size * 1.2)  # Line spacing

                png_exists = watermark_png_path and os.path.exists(
                    watermark_png_path)

                if png_exists:
                    if png_location == "topright":
                        # PNG at topright, text cycles: bottomright -> bottomleft -> bottomright -> bottomleft (2 positions, 120s cycle)
                        text_filter = (
                            f"drawtext=text='{final_text}':"
                            f"fontsize={font_size}:"
                            f"fontcolor=white@0.95:"
                            f"box=1:boxcolor=black@0.8:boxborderw=10:"
                            f"line_spacing={line_spacing}:"
                            f"x='if(lt(mod(t,120),60),w-text_w-{margin},{margin})':"
                            f"y='h-text_h-{margin}'"
                        )
                    else:
                        # PNG not at topright, text cycles: topright -> bottomright -> bottomleft (3 positions, 180s cycle) - never use topleft
                        text_filter = (
                            f"drawtext=text='{final_text}':"
                            f"fontsize={font_size}:"
                            f"fontcolor=white@0.95:"
                            f"box=1:boxcolor=black@0.8:boxborderw=10:"
                            f"line_spacing={line_spacing}:"
                            f"x='if(lt(mod(t,180),60),w-text_w-{margin},if(lt(mod(t,180),120),w-text_w-{margin},{margin}))':"
                            f"y='if(lt(mod(t,180),60),{margin*2},if(lt(mod(t,180),120),h-text_h-{margin},h-text_h-{margin}))'"
                        )
                else:
                    # No PNG, text cycles: topright -> bottomright -> bottomleft (3 positions, 180s cycle) - never use topleft
                    text_filter = (
                        f"drawtext=text='{final_text}':"
                        f"fontsize={font_size}:"
                        f"fontcolor=white@0.95:"
                        f"box=1:boxcolor=black@0.8:boxborderw=10:"
                        f"line_spacing={line_spacing}:"
                        f"x='if(lt(mod(t,180),60),w-text_w-{margin},if(lt(mod(t,180),120),w-text_w-{margin},{margin}))':"
                        f"y='if(lt(mod(t,180),60),{margin*2},if(lt(mod(t,180),120),h-text_h-{margin},h-text_h-{margin}))'"
                    )

                filter_parts.append(f"{overlay_inputs}{text_filter}[txt]")
                overlay_inputs = "[txt]"

            # Add PNG watermark if provided
            if watermark_png_path and os.path.exists(watermark_png_path):
                max_size = min(width // 4, height // 4)
                margin = 8

                # PNG position based on selected location
                if png_location == "topleft":
                    png_x, png_y = margin, margin
                elif png_location == "topright":
                    png_x, png_y = f"W-w-{margin}", margin
                elif png_location == "bottomleft":
                    png_x, png_y = margin, f"H-h-{margin}"
                else:  # bottomright (default)
                    png_x, png_y = f"W-w-{margin}", f"H-h-{margin}"

                filter_parts.append(
                    f"[1:v]scale=-1:{max_size}:force_original_aspect_ratio=decrease[wm]"
                )
                filter_parts.append(
                    f"{overlay_inputs}[wm]overlay={png_x}:{png_y}[final]"
                )
                overlay_inputs = "[final]"

            # Get original video bitrate for size matching
            original_bitrate = video_info.get('bitrate', 0)
            if original_bitrate > 0:
                # Use original bitrate but cap it for reasonable size
                target_bitrate = min(original_bitrate, 2000000)  # Max 2Mbps
            else:
                # Estimate bitrate based on file size and duration
                file_size = os.path.getsize(input_path)
                duration = video_info.get('duration', 1)
                if duration > 0:
                    estimated_bitrate = int((file_size * 8) / duration *
                                            0.85)  # 85% for video track
                    target_bitrate = min(estimated_bitrate, 2000000)
                else:
                    target_bitrate = 1000000  # 1Mbps fallback

            target_bitrate_str = f"{target_bitrate//1000}k"

            # Build command optimized for maintaining original file size
            if watermark_png_path and os.path.exists(watermark_png_path):
                if filter_parts:
                    cmd = [
                        'ffmpeg',
                        '-i',
                        input_path,
                        '-i',
                        watermark_png_path,
                        '-filter_complex',
                        ';'.join(filter_parts),
                        '-map',
                        overlay_inputs,
                        '-map',
                        '0:a?',
                        '-c:v',
                        'libx264',
                        '-preset',
                        'ultrafast',
                        '-crf',
                        '28',  # Higher CRF for smaller size
                        '-c:a',
                        'copy',  # Copy audio without re-encoding
                        '-r',
                        str(fps),
                        '-b:v',
                        target_bitrate_str,  # Match original bitrate
                        '-maxrate',
                        f"{int(target_bitrate*1.2)//1000}k",
                        '-bufsize',
                        f"{int(target_bitrate*2)//1000}k",
                        '-profile:v',
                        'main',
                        '-level',
                        '3.1',
                        '-map_metadata',
                        '0',
                        '-movflags',
                        '+faststart',
                        '-threads',
                        '0',
                        '-y',
                        output_path
                    ]
                else:
                    # Only PNG watermark, no text
                    margin = 15
                    cmd = [
                        'ffmpeg', '-i', input_path, '-i', watermark_png_path,
                        '-filter_complex',
                        f"[1:v]scale=-1:{min(width // 4, height // 4)}:force_original_aspect_ratio=decrease[wm];[0:v][wm]overlay=W-w-{margin}:{margin}[final]",
                        '-map', '[final]', '-map', '0:a?', '-c:v', 'libx264',
                        '-preset', 'ultrafast', '-crf', '28', '-c:a', 'copy',
                        '-r',
                        str(fps), '-b:v', target_bitrate_str, '-maxrate',
                        f"{int(target_bitrate*1.2)//1000}k", '-bufsize',
                        f"{int(target_bitrate*2)//1000}k", '-profile:v',
                        'main', '-level', '3.1', '-map_metadata', '0',
                        '-movflags', '+faststart', '-threads', '0', '-y',
                        output_path
                    ]
            else:
                if filter_parts:
                    # Extract the filter properly - get everything after the input reference
                    text_filter_only = filter_parts[0].replace(
                        f"{overlay_inputs}", "").replace("[txt]", "")
                    cmd = [
                        'ffmpeg',
                        '-i',
                        input_path,
                        '-vf',
                        text_filter_only,
                        '-c:v',
                        'libx264',
                        '-preset',
                        'ultrafast',
                        '-crf',
                        '28',  # Higher CRF for smaller size
                        '-c:a',
                        'copy',
                        '-r',
                        str(fps),
                        '-b:v',
                        target_bitrate_str,  # Match original bitrate
                        '-maxrate',
                        f"{int(target_bitrate*1.2)//1000}k",
                        '-bufsize',
                        f"{int(target_bitrate*2)//1000}k",
                        '-profile:v',
                        'main',
                        '-level',
                        '3.1',
                        '-map_metadata',
                        '0',
                        '-movflags',
                        '+faststart',
                        '-threads',
                        '0',
                        '-y',
                        output_path
                    ]
                else:
                    # No watermarks at all, just copy streams for maximum speed and size preservation
                    cmd = [
                        'ffmpeg',
                        '-i',
                        input_path,
                        '-c:v',
                        'copy',  # Copy video stream without re-encoding
                        '-c:a',
                        'copy',  # Copy audio stream without re-encoding
                        '-map_metadata',
                        '0',
                        '-movflags',
                        '+faststart',
                        '-y',
                        output_path
                    ]

            # Execute with enhanced progress monitoring
            if progress_callback:
                process = subprocess.Popen(cmd,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)

                frame_count = 0
                process_start_time = time.time()
                last_progress_update = process_start_time
                fps_samples = []

                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        if 'frame=' in output:
                            try:
                                current_time = time.time()
                                frame_match = output.split(
                                    'frame=')[1].split()[0]
                                frame_count = int(frame_match)

                                # Calculate processing FPS
                                elapsed_time = current_time - process_start_time
                                if elapsed_time > 0:
                                    processing_fps = frame_count / elapsed_time
                                    fps_samples.append(processing_fps)

                                    # Keep only last 10 samples for smoothing
                                    if len(fps_samples) > 10:
                                        fps_samples.pop(0)

                                    avg_fps = sum(fps_samples) / len(
                                        fps_samples)
                                else:
                                    avg_fps = 0

                                # Calculate time estimates
                                if avg_fps > 0 and total_frames > 0:
                                    remaining_frames = max(
                                        0, total_frames - frame_count)
                                    eta_seconds = remaining_frames / avg_fps

                                    # Format ETA
                                    if eta_seconds > 3600:
                                        eta_mins = int(eta_seconds // 3600)
                                        eta_secs = int((eta_seconds % 3600) // 60)
                                        eta_str = f"{eta_mins}h{eta_secs}m"
                                    elif eta_seconds > 60:
                                        eta_mins = int(eta_seconds // 60)
                                        eta_secs = int(eta_seconds % 60)
                                        eta_str = f"{eta_mins}m{eta_secs}s"
                                    else:
                                        eta_str = f"{int(eta_seconds)}s"
                                else:
                                    eta_str = "calculating..."

                                # Format elapsed time
                                if elapsed_time > 3600:
                                    elapsed_mins = int(elapsed_time // 3600)
                                    elapsed_secs = int((elapsed_time % 3600) // 60)
                                    elapsed_str = f"{elapsed_mins}h{elapsed_secs}m"
                                elif elapsed_time > 60:
                                    elapsed_mins = int(elapsed_time // 60)
                                    elapsed_secs = int(elapsed_time % 60)
                                    elapsed_str = f"{elapsed_mins}m{elapsed_secs}s"
                                else:
                                    elapsed_str = f"{int(elapsed_time)}s"

                                # Update progress every 3 seconds for flood protection
                                if (current_time - last_progress_update
                                        >= 3.0) or (frame_count
                                                    >= total_frames):
                                    # Create detailed progress info
                                    progress_info = {
                                        'frame_count': frame_count,
                                        'total_frames': total_frames,
                                        'processing_fps': avg_fps,
                                        'elapsed': elapsed_str,
                                        'eta': eta_str,
                                        'stage': 'watermarking'
                                    }

                                    await progress_callback(
                                        frame_count, total_frames,
                                        progress_info)
                                    last_progress_update = current_time

                            except Exception as e:
                                logger.warning(
                                    f"Error parsing FFmpeg output: {e}")
                                pass

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, cmd)
            else:
                result = subprocess.run(cmd,
                                        check=True,
                                        capture_output=True,
                                        text=True)

            logger.info(
                "Successfully added combined watermarks with metadata preservation"
            )

        except Exception as e:
            logger.error(f"Combined watermark error: {e}")
            shutil.copy2(input_path, output_path)

    async def add_intro_with_normalization(self,
                                           intro_path: str,
                                           video_path: str,
                                           output_path: str,
                                           progress_callback=None):
        """Add intro with normalization tracking - normalizes every time"""
        import subprocess

        # Initialize variables at the top to prevent UnboundLocalError
        temp_intro = None
        temp_main = None
        list_file = None

        try:
            # Get video properties
            intro_info = await self.get_video_info(intro_path)
            main_info = await self.get_video_info(video_path)

            # Use consistent target specifications with improved quality
            target_fps = 30  # Fixed FPS for compatibility
            target_width = 1280
            target_height = 720

            # Create temp files for normalized videos
            temp_intro = f"temp_intro_normalized_{int(time.time())}.mp4"
            temp_main = f"temp_main_normalized_{int(time.time())}.mp4"

            # Stage 1: Normalize intro (0-25%)
            if progress_callback:
                await progress_callback(0, 100, "Starting intro normalization")

            intro_normalize_cmd = [
                'ffmpeg',
                '-i',
                intro_path,
                '-vf',
                f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,fps={target_fps}',
                '-c:v',
                'libx264',
                '-preset',
                'ultrafast',
                '-crf',
                '32',  # Higher CRF for much smaller size
                '-c:a',
                'copy',  # Copy audio for speed
                '-pix_fmt',
                'yuv420p',
                '-b:v',
                '800k',  # Lower bitrate for intro
                '-maxrate',
                '1M',
                '-bufsize',
                '2M',  # Much lower bitrate for smaller files
                '-profile:v',
                'main',
                '-level',
                '3.1',
                '-movflags',
                '+faststart',
                '-threads',
                '0',
                '-f',
                'mp4',
                '-y',
                temp_intro
            ]

            # Track intro normalization progress
            if progress_callback:
                process = subprocess.Popen(intro_normalize_cmd,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)

                intro_frame_count = 0
                intro_total_frames = int(target_fps *
                                         intro_info.get('duration', 0))
                last_intro_update = time.time()
                intro_start_time = time.time()

                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        if 'frame=' in output:
                            try:
                                current_time = time.time()
                                frame_match = output.split(
                                    'frame=')[1].split()[0]
                                intro_frame_count = int(frame_match)

                                # Calculate processing metrics
                                elapsed_time = current_time - intro_start_time
                                if elapsed_time > 0:
                                    processing_fps = intro_frame_count / elapsed_time
                                else:
                                    processing_fps = 0

                                # Calculate ETA
                                if processing_fps > 0 and intro_total_frames > 0:
                                    remaining_frames = max(
                                        0,
                                        intro_total_frames - intro_frame_count)
                                    eta_seconds = remaining_frames / processing_fps

                                    if eta_seconds > 60:
                                        eta_mins = int(eta_seconds // 60)
                                        eta_secs = int(eta_seconds % 60)
                                        eta_str = f"{eta_mins}m{eta_secs}s"
                                    else:
                                        eta_str = f"{int(eta_seconds)}s"
                                else:
                                    eta_str = "calculating..."

                                # Format elapsed
                                if elapsed_time > 60:
                                    elapsed_mins = int(elapsed_time // 60)
                                    elapsed_secs = int(elapsed_time % 60)
                                    elapsed_str = f"{elapsed_mins}m{elapsed_secs}s"
                                else:
                                    elapsed_str = f"{int(elapsed_time)}s"

                                # Update progress every 5 seconds for flood protection
                                if current_time - last_intro_update >= 5.0:
                                    if intro_total_frames > 0:
                                        frame_percent = (
                                            intro_frame_count /
                                            intro_total_frames
                                        ) * 25  # 25% for intro
                                        step_detail = f"Normalizing intro: {intro_frame_count}/{intro_total_frames} @ {processing_fps:.1f}fps"
                                        await progress_callback(
                                            frame_percent,
                                            100,
                                            step_detail,
                                            elapsed=elapsed_str,
                                            eta=eta_str)
                                    else:
                                        await progress_callback(
                                            12,
                                            100,
                                            f"Normalizing intro: {intro_frame_count} frames @ {processing_fps:.1f}fps",
                                            elapsed=elapsed_str)
                                    last_intro_update = current_time
                            except Exception as e:
                                logger.warning(
                                    f"Error parsing intro progress: {e}")
                                pass

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, intro_normalize_cmd)
            else:
                result = subprocess.run(intro_normalize_cmd,
                                        check=True,
                                        capture_output=True,
                                        text=True)

            logger.info("Intro normalization completed")

            # Stage 2: Normalize main video (25-70%)
            if progress_callback:
                await progress_callback(25, 100,
                                        "Starting main video normalization")

            # Get original bitrate for main video to maintain size
            original_main_bitrate = main_info.get('bitrate', 0)
            if original_main_bitrate > 0:
                main_target_bitrate = min(original_main_bitrate,
                                          1500000)  # Max 1.5Mbps
            else:
                # Estimate from file size
                main_file_size = os.path.getsize(video_path)
                main_duration = main_info.get('duration', 1)
                if main_duration > 0:
                    estimated_main_bitrate = int(
                        (main_file_size * 8) / main_duration * 0.85)
                    main_target_bitrate = min(estimated_main_bitrate, 1500000)
                else:
                    main_target_bitrate = 1000000

            main_target_bitrate_str = f"{main_target_bitrate//1000}k"

            main_normalize_cmd = [
                'ffmpeg',
                '-i',
                video_path,
                '-vf',
                f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,fps={target_fps}',
                '-c:v',
                'libx264',
                '-preset',
                'ultrafast',
                '-crf',
                '30',  # Higher CRF for smaller size
                '-c:a',
                'copy',  # Copy audio for speed
                '-pix_fmt',
                'yuv420p',
                '-b:v',
                main_target_bitrate_str,  # Use calculated bitrate
                '-maxrate',
                f"{int(main_target_bitrate*1.1)//1000}k",
                '-bufsize',
                f"{int(main_target_bitrate*2)//1000}k",
                '-profile:v',
                'main',
                '-level',
                '3.1',
                '-movflags',
                '+faststart',
                '-threads',
                '0',
                '-f',
                'mp4',
                '-y',
                temp_main
            ]

            # Track main video normalization progress
            if progress_callback:
                process = subprocess.Popen(main_normalize_cmd,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)

                main_frame_count = 0
                main_total_frames = int(target_fps *
                                        main_info.get('duration', 0))
                last_main_update = time.time()
                main_start_time = time.time()

                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        if 'frame=' in output:
                            try:
                                current_time = time.time()
                                frame_match = output.split(
                                    'frame=')[1].split()[0]
                                main_frame_count = int(frame_match)

                                # Calculate processing metrics
                                elapsed_time = current_time - main_start_time
                                if elapsed_time > 0:
                                    processing_fps = main_frame_count / elapsed_time
                                else:
                                    processing_fps = 0

                                # Calculate ETA
                                if processing_fps > 0 and main_total_frames > 0:
                                    remaining_frames = max(
                                        0,
                                        main_total_frames - main_frame_count)
                                    eta_seconds = remaining_frames / processing_fps

                                    if eta_seconds > 60:
                                        eta_mins = int(eta_seconds // 60)
                                        eta_secs = int(eta_seconds % 60)
                                        eta_str = f"{eta_mins}m{eta_secs}s"
                                    else:
                                        eta_str = f"{int(eta_seconds)}s"
                                else:
                                    eta_str = "calculating..."

                                # Format elapsed
                                if elapsed_time > 60:
                                    elapsed_mins = int(elapsed_time // 60)
                                    elapsed_secs = int(elapsed_time % 60)
                                    elapsed_str = f"{elapsed_mins}m{elapsed_secs}s"
                                else:
                                    elapsed_str = f"{int(elapsed_time)}s"

                                # Update progress every 5 seconds for flood protection
                                if current_time - last_main_update >= 5.0:
                                    if main_total_frames > 0:
                                        frame_percent = (
                                            main_frame_count /
                                            main_total_frames
                                        ) * 45  # 45% for main video (25-70%)
                                        overall_percent = 25 + frame_percent
                                        step_detail = f"Normalizing main: {main_frame_count}/{main_total_frames} @ {processing_fps:.1f}fps"
                                        await progress_callback(
                                            overall_percent,
                                            100,
                                            step_detail,
                                            elapsed=elapsed_str,
                                            eta=eta_str)
                                    else:
                                        await progress_callback(
                                            47,
                                            100,
                                            f"Normalizing main: {main_frame_count} frames @ {processing_fps:.1f}fps",
                                            elapsed=elapsed_str)
                                    last_main_update = current_time
                            except Exception as e:
                                logger.warning(
                                    f"Error parsing main progress: {e}")
                                pass

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, main_normalize_cmd)
            else:
                result = subprocess.run(main_normalize_cmd,
                                        check=True,
                                        capture_output=True,
                                        text=True)

            logger.info("Main video normalization completed")

            # Stage 3: Concatenate videos (70-100%)
            if progress_callback:
                await progress_callback(70, 100,
                                        "Concatenating normalized videos")

            # Create file list for concatenation
            list_file = f"temp_list_{int(time.time())}.txt"
            with open(list_file, 'w') as f:
                f.write(f"file '{os.path.abspath(temp_intro)}'\n")
                f.write(f"file '{os.path.abspath(temp_main)}'\n")

            # Use ultra-fast concatenation settings
            cmd_concat = [
                'ffmpeg',
                '-f',
                'concat',
                '-safe',
                '0',
                '-i',
                list_file,
                '-c',
                'copy',  # Copy streams without re-encoding for maximum speed
                '-avoid_negative_ts',
                'make_zero',
                '-fflags',
                '+genpts',
                '-movflags',
                '+faststart',
                '-y',
                output_path
            ]

            if progress_callback:
                process = subprocess.Popen(cmd_concat,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)
                concat_start = time.time()
                last_concat_update = concat_start

                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        current_time = time.time()
                        # Update every 10 seconds
                        if current_time - last_concat_update >= 10.0:
                            elapsed = current_time - concat_start
                            concat_percent = 70 + min(
                                30, (elapsed / 10) *
                                30)  # Progress from 70% to 100%
                            await progress_callback(
                                concat_percent, 100,
                                f"Concatenating videos ({elapsed:.1f}s)")
                            last_concat_update = current_time

                process.wait()
                if process.returncode != 0:
                    stderr_output = process.stderr.read(
                    ) if process.stderr else "Unknown error"
                    logger.error(
                        f"Concatenation failed with stderr: {stderr_output}")
                    raise subprocess.CalledProcessError(
                        process.returncode, cmd_concat, stderr_output)
            else:
                result = subprocess.run(cmd_concat,
                                        check=True,
                                        capture_output=True,
                                        text=True)

            if progress_callback:
                await progress_callback(
                    100, 100, "Intro and main video processing completed")

            logger.info("Successfully concatenated intro with main video")

        except subprocess.CalledProcessError as e:
            logger.error(
                f"FFmpeg error during processing: {e.stderr if hasattr(e, 'stderr') else str(e)}"
            )
            raise Exception(
                f"Video processing failed: FFmpeg error - {str(e)}")
        except Exception as e:
            logger.error(
                f"Video processing with intro normalization failed: {e}")
            raise Exception(f"Intro processing failed: {str(e)}")
        finally:
            # Clean up all temp files
            for temp_file in [temp_intro, temp_main, list_file]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.info(f"Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up {temp_file}: {e}")

            # Kill any remaining FFmpeg processes
            try:
                import subprocess
                subprocess.run(['pkill', '-f', 'ffmpeg'], check=False, capture_output=True)
            except:
                pass

    async def process_video_with_metadata(self,
                                          message: Message,
                                          user_id: int,
                                          video_path: str,
                                          watermark_text: str,
                                          original_caption: str,
                                          video_num: int = None,
                                          png_location: str = "topright"):
        """Process video with smooth progress tracking throughout all stages"""
        progress_msg = None
        try:
            session = self.user_sessions.get(user_id, {})
            if session.get('stopped'):
                return

            # Initialize unified progress tracking
            video_counter = f"[{video_num}/{self.user_sessions[user_id].get('total_videos', 1)}] " if video_num else ""
            progress_msg = await self.create_progress_message(
                message, f"🚀 **Initializing video processing...**", video_num,
                self.user_sessions[user_id].get('total_videos', 1)
                if video_num else None)

            start_time = time.time()
            last_update_time = 0

            async def update_timed_progress(stage: str,
                                            percent: float,
                                            step_detail: str = "",
                                            **kwargs):
                nonlocal last_update_time
                current_time = time.time()

                # Check if stopped
                if self.user_sessions.get(user_id, {}).get('stopped'):
                    raise Exception("Process was stopped by user")

                # Ensure percent is within valid bounds
                percent = max(0, min(100, percent))

                # Ensure 10-second delay between updates for flood protection
                if current_time - last_update_time < 10 and percent < 100:
                    return

                # Always update if it's completion (100%)
                if percent >= 100 or current_time - last_update_time >= 10:
                    # Format elapsed time
                    total_elapsed = current_time - start_time
                    if total_elapsed > 3600:
                        elapsed_str = f"{int(total_elapsed//3600)}h {int((total_elapsed%3600)//60)}m"
                    elif total_elapsed > 60:
                        elapsed_str = f"{int(total_elapsed//60)}m {int(total_elapsed%60)}s"
                    else:
                        elapsed_str = f"{int(total_elapsed)}s"

                    # Build details
                    details = {
                        'elapsed': elapsed_str,
                        'processing_step': step_detail,
                        **kwargs
                    }

                    await self.update_progress_smooth(progress_msg, stage,
                                                      percent, details,
                                                      video_counter, True)

                    last_update_time = current_time

            # Stage 1: Download video if needed (0-25%)
            if not os.path.exists(video_path):
                await update_timed_progress("downloading", 0,
                                            "Starting download")

                session = self.user_sessions[user_id]
                file_id = session['video_file_id']

                download_start_time = time.time()
                speed_samples = []
                last_current = 0
                last_download_update = download_start_time

                async def smooth_download_progress(current, total):
                    nonlocal speed_samples, last_current, last_download_update
                    current_time = time.time()

                    if session.get('stopped'):
                        raise Exception("Process was stopped by user")

                    # Calculate download metrics
                    download_percent = (
                        current /
                        total) * 25 if total > 0 else 0  # 0-25% for download
                    current_mb = current / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    elapsed = current_time - download_start_time

                    # Speed calculation
                    if elapsed > 0:
                        bytes_diff = current - last_current
                        time_diff = current_time - last_download_update
                        instant_speed = bytes_diff / time_diff if time_diff > 0 else 0

                        speed_samples.append(instant_speed)
                        if len(speed_samples) > 10:
                            speed_samples.pop(0)

                        avg_speed = sum(speed_samples) / len(
                            speed_samples) if speed_samples else 0
                        speed_mbps = avg_speed / (1024 * 1024)

                        # ETA calculation
                        if avg_speed > 0 and current < total:
                            remaining_bytes = total - current
                            eta_seconds = remaining_bytes / avg_speed
                            if eta_seconds > 60:
                                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                            else:
                                eta_str = f"{int(eta_seconds)}s"
                        else:
                            eta_str = "calculating..."
                    else:
                        speed_mbps = 0
                        eta_str = "calculating..."

                    # Update every 10 seconds or on completion
                    if (current_time - last_download_update
                            >= 10) or (current == total):
                        await update_timed_progress(
                            "downloading",
                            download_percent,
                            f"Downloading: {current_mb:.1f}/{total_mb:.1f} MB",
                            current_mb=current_mb,
                            total_mb=total_mb,
                            speed_mbps=speed_mbps,
                            eta=eta_str)
                        last_download_update = current_time

                    last_current = current

                try:
                    await self.app.download_media(
                        file_id,
                        file_name=video_path,
                        progress=smooth_download_progress)
                    await update_timed_progress("downloading", 25,
                                                "Download completed")
                except Exception as e:
                    if "stopped by user" in str(e):
                        raise
                    else:
                        logger.error(f"Download error: {e}")
                        raise Exception(f"Download failed: {str(e)}")

            # Stage 2: Apply watermarks (25-60%)
            await update_timed_progress("watermarking", 25,
                                        "Preparing watermarks")
            watermarked_path = video_path.replace('.mp4', '_watermarked.mp4')
            watermark_png_path = f"persistent_watermarks/watermark_{user_id}.png"

            last_watermark_update = time.time()

            async def watermark_progress(current_frame,
                                         total_frames,
                                         progress_info=None):
                nonlocal last_watermark_update
                current_time = time.time()

                if total_frames > 0:
                    frame_ratio = min(1.0, current_frame / total_frames)
                    frame_percent = frame_ratio * 35  # 35% for watermarking
                    total_percent = max(25, min(60, 25 + frame_percent))

                    # Update every 3 seconds or on completion
                    if (current_time - last_watermark_update
                            >= 3) or (current_frame >= total_frames):
                        # Build detailed step description
                        if progress_info:
                            processing_fps = progress_info.get(
                                'processing_fps', 0)
                            elapsed = progress_info.get('elapsed', '0s')
                            eta = progress_info.get('eta', 'calculating...')

                            step_detail = f"Watermarking: {current_frame}/{total_frames} frames @ {processing_fps:.1f}fps"
                        else:
                            step_detail = f"Watermarking: {current_frame}/{total_frames} frames"

                        # Enhanced details for progress display
                        details = {
                            'frame_progress':
                            f"{current_frame}/{total_frames}",
                            'processing_fps':
                            progress_info.get('processing_fps', 0)
                            if progress_info else 0,
                            'elapsed':
                            progress_info.get('elapsed', '')
                            if progress_info else '',
                            'eta':
                            progress_info.get('eta', '')
                            if progress_info else ''
                        }

                        await update_timed_progress(
                            "watermarking",
                            total_percent, step_detail,
                            **details)
                        last_watermark_update = current_time

            await self.add_combined_watermarks(video_path, watermarked_path,
                                               watermark_text,
                                               watermark_png_path,
                                               png_location,
                                               watermark_progress)
            await update_timed_progress("watermarking", 60,
                                        "Watermarking completed")

            # Stage 3: Add intro if exists (60-85%)
            final_path = watermarked_path
            intro_path = f"persistent_intros/intro_{user_id}.mp4"
            if os.path.exists(intro_path):
                try:
                    intro_info = await self.get_video_info(intro_path)
                    if intro_info.get('duration', 0) > 0 and intro_info.get(
                            'width', 0) > 0:
                        await update_timed_progress("processing", 60,
                                                    "Validating intro video")
                        final_with_intro = video_path.replace(
                            '.mp4', '_with_intro.mp4')

                        last_intro_update = time.time()

                        async def intro_progress(percent, total, step_desc, elapsed=None, eta=None):
                            nonlocal last_intro_update
                            current_time = time.time()

                            # Map intro progress to 60-85% range
                            intro_ratio = max(0, min(100, percent)) / 100
                            intro_percent = max(60, min(85, 60 + intro_ratio * 25))

                            # Update every 10 seconds or on completion
                            if (current_time - last_intro_update
                                    >= 10) or (percent >= 100):
                                await update_timed_progress(
                                    "processing", intro_percent, step_desc)
                                last_intro_update = current_time

                        try:
                            await self.add_intro_with_normalization(
                                intro_path, watermarked_path, final_with_intro,
                                intro_progress)

                            if os.path.exists(
                                    final_with_intro) and os.path.getsize(
                                        final_with_intro) > 0:
                                final_path = final_with_intro
                                await update_timed_progress(
                                    "processing", 85,
                                    "Intro addition completed")
                            else:
                                await update_timed_progress(
                                    "processing", 85,
                                    "Intro failed, using watermarked video")
                        except Exception as intro_error:
                            logger.error(
                                f"Intro processing failed: {intro_error}")
                            await update_timed_progress(
                                "processing", 85,
                                f"Intro error: {str(intro_error)[:30]}...")
                    else:
                        await update_timed_progress(
                            "processing", 85, "Skipping corrupted intro")
                except Exception as e:
                    logger.error(f"Error validating intro: {e}")
                    await update_timed_progress("processing", 85,
                                                "Intro validation failed")
            else:
                await update_timed_progress("processing", 85,
                                            "No intro found, proceeding")

            # Stage 4: Prepare caption (85-90%)
            await update_timed_progress("finalizing", 85, "Preparing captions")
            permanent_caption_path = f"persistent_captions/caption_{user_id}.txt"
            permanent_caption = ""
            if os.path.exists(permanent_caption_path):
                try:
                    with open(permanent_caption_path, 'r') as f:
                        permanent_caption = f.read().strip()
                except Exception as e:
                    logger.error(f"Error reading permanent caption: {e}")

            combined_caption = original_caption
            if permanent_caption:
                combined_caption = f"{original_caption}\n\n{permanent_caption}"

            await update_timed_progress("finalizing", 90,
                                        "Caption preparation completed")

            # Stage 5: Generate thumbnail if not exists (85-90%)
            await update_timed_progress("finalizing", 87, "Checking thumbnail")

            thumbnail_path = f"persistent_thumbnails/thumbnail_{user_id}.jpg"
            auto_thumbnail_path = f"temp_{user_id}/auto_thumbnail.jpg"

            # Use custom thumbnail if exists, otherwise generate from video
            thumbnail = None
            if os.path.exists(thumbnail_path):
                thumbnail = thumbnail_path
                await update_timed_progress("finalizing", 89,
                                            "Using custom thumbnail")
            else:
                # Generate automatic thumbnail from final video
                await update_timed_progress("finalizing", 88,
                                            "Generating thumbnail from video")
                if await self.generate_video_thumbnail(final_path,
                                                       auto_thumbnail_path):
                    thumbnail = auto_thumbnail_path
                    await update_timed_progress("finalizing", 89,
                                                "Auto thumbnail generated")
                else:
                    await update_timed_progress("finalizing", 89,
                                                "No thumbnail available")

            # Stage 6: Upload with integrated progress tracking (85-100%)
            await update_timed_progress("uploading", 85, "Starting upload")

            session = self.user_sessions[user_id]
            video_duration = session.get('video_duration', 0)
            video_width = session.get('video_width', 1280)
            video_height = session.get('video_height', 720)

            # Get file size for upload progress and overall ETA calculation
            upload_file_size = os.path.getsize(final_path)
            total_process_start = start_time  # Use overall process start time
            last_upload_update = time.time()
            upload_speed_samples = []

            async def upload_progress_callback(current, total):
                nonlocal last_upload_update, upload_speed_samples
                current_time = time.time()

                if session.get('stopped'):
                    raise Exception("Process was stopped by user")

                # Calculate upload metrics with integrated progress (85-100% = 15% total)
                upload_ratio = (current / total) if total > 0 else 0
                upload_percent = max(85, min(100, 85 + upload_ratio * 15))
                current_mb = current / (1024 * 1024)
                total_mb = total / (1024 * 1024)

                # Calculate upload speed
                upload_elapsed = current_time - last_upload_update if last_upload_update > 0 else 1
                if upload_elapsed > 0:
                    current_speed = current / upload_elapsed if last_upload_update > 0 else 0
                    upload_speed_samples.append(current_speed)
                    if len(upload_speed_samples) > 5:
                        upload_speed_samples.pop(0)

                    avg_speed = sum(upload_speed_samples) / len(
                        upload_speed_samples) if upload_speed_samples else 0
                    speed_mbps = avg_speed / (1024 * 1024)

                    # Calculate remaining upload time only
                    if avg_speed > 0 and current < total:
                        remaining_bytes = total - current
                        upload_eta_seconds = remaining_bytes / avg_speed
                        if upload_eta_seconds > 60:
                            eta_str = f"{int(upload_eta_seconds//60)}m{int(upload_eta_seconds%60)}s"
                        else:
                            eta_str = f"{int(upload_eta_seconds)}s"
                    else:
                        eta_str = "finishing..."
                else:
                    speed_mbps = 0
                    eta_str = "calculating..."

                # Overall elapsed time from process start
                total_elapsed = current_time - total_process_start
                if total_elapsed > 60:
                    elapsed_str = f"{int(total_elapsed//60)}m{int(total_elapsed%60)}s"
                else:
                    elapsed_str = f"{int(total_elapsed)}s"

                # Update every 2 seconds for more responsive upload tracking
                if (current_time - last_upload_update >= 2) or (current
                                                                == total):
                    await update_timed_progress(
                        "uploading",
                        upload_percent,
                        f"Uploading: {current_mb:.2f}/{total_mb:.2f} MB",
                        current_mb=current_mb,
                        total_mb=total_mb,
                        speed_mbps=max(0.01, speed_mbps),
                        eta=eta_str,
                        elapsed=elapsed_str)
                    last_upload_update = current_time

            try:
                await self.app.send_video(
                    chat_id=user_id,
                    video=final_path,
                    caption=combined_caption,
                    duration=int(video_duration)
                    if video_duration > 0 else None,
                    width=int(video_width) if video_width > 0 else None,
                    height=int(video_height) if video_height > 0 else None,
                    thumb=thumbnail,
                    supports_streaming=True,
                    progress=upload_progress_callback)
            except Exception as upload_error:
                logger.error(f"Upload error: {upload_error}")
                # Fallback upload without progress
                await self.app.send_video(chat_id=user_id,
                                          video=final_path,
                                          caption=combined_caption)

            # Calculate total processing time
            total_time = time.time() - start_time
            if total_time > 3600:
                time_str = f"{int(total_time//3600)}h {int((total_time%3600)//60)}m {int(total_time%60)}s"
            elif total_time > 60:
                time_str = f"{int(total_time//60)}m {int(total_time%60)}s"
            else:
                time_str = f"{int(total_time)}s"

            await update_timed_progress("completed", 100,
                                        f"Processing completed in {time_str}")

            # IMPROVED CLEANUP: Always delete progress message after completion
            if progress_msg:
                try:
                    await asyncio.sleep(3)  # Wait 3 seconds to show completion
                    await progress_msg.delete()
                    logger.info("Progress message deleted successfully")
                except Exception as e:
                    logger.warning(f"Could not delete progress message: {e}")

            # Clean up temp files
            user_dir = f"temp_{user_id}"
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)

            for temp_file in [
                    video_path, watermarked_path,
                    final_path if final_path != watermarked_path else None
            ]:
                if temp_file and temp_file != final_path and os.path.exists(
                        temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

            if not video_num:  # Single video processing
                # Only reset processing flag, keep other session data
                if user_id in self.user_sessions and 'processing' in self.user_sessions[user_id]:
                    self.user_sessions[user_id].pop('processing', None)
                await message.reply_text(
                    f"🎉 Video processing completed successfully!\n⏱️ **Total time:** {time_str}")

        except Exception as e:
            error_msg = str(e)
            if "stopped by user" in error_msg:
                await message.reply_text("🛑 **Processing stopped by user**")
            else:
                await message.reply_text(
                    f"❌ Error processing video: {error_msg}")
            logger.error(f"Video processing error for user {user_id}: {e}")

            # IMPROVED ERROR CLEANUP: Always delete progress message on error
            if progress_msg:
                try:
                    await asyncio.sleep(2)
                    await progress_msg.delete()
                    logger.info("Progress message deleted after error")
                except Exception as e:
                    logger.warning(
                        f"Could not delete progress message after error: {e}")
        finally:
            # Always ensure processing flag is cleared for this user
            if user_id in self.user_sessions and 'processing' in self.user_sessions[user_id]:
                self.user_sessions[user_id].pop('processing', None)

            # Cleanup temp files for this user
            user_dir = f"temp_{user_id}"
            if os.path.exists(user_dir):
                try:
                    shutil.rmtree(user_dir)
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up temp dir for user {user_id}: {cleanup_error}")

    # Command handlers
    async def start_handler(self, client: Client, message: Message):
        """Start command handler"""
        user_id = message.from_user.id


        # Clear any existing stopped flag and reset session
        if user_id in self.user_sessions:
            self.user_sessions[user_id] = {}

        system_info = self.get_system_usage()
        is_admin = user_id in self.admin_users

        admin_commands = ""
        if is_admin:
            admin_commands = """
🔧 **Admin Commands:**
• /adduser <user_id> - Add user to admin list
• /removeuser <user_id> - Remove user from admin list
• /listadmins - List all admin users
• /stats - Get detailed bot statistics
• /cleanup - Clean all temp and cache files

"""

        welcome_text = f"""
🎬 **Watermark Bot By ZeroTrace** 🎬
{system_info}
👤 **User ID:** {user_id} {'🔧 (Admin)' if is_admin else ''}

I can handle videos up to 1GB with advanced features!

✅ **Available Commands:**
• /setintro - Set intro video for future use
• /setwatermark - Set PNG watermark image for future use
• /setthumbnail - Set thumbnail for watermarked videos
• /addcaption - Add permanent caption (saved until removed)
• /convert - Convert MP4 file format with watermarks
• /removeintro - Remove saved intro
• /removewatermark - Remove saved watermark
• /removethumbnail - Remove saved thumbnail
• /removecaption - Remove saved permanent caption
• /status - Check saved settings
• /bulk - Start bulk processing mode
• /queue - Check bulk queue status
• /stop - Stop all processes and clean up

{admin_commands}📋 **How to use:**
1. (Optional) Set intro/watermark/caption using commands above
2. Send me a video to watermark (up to 1GB!) OR use /convert for file format
3. If PNG watermark is set, choose location (topleft/topright/bottomleft/bottomright)
4. Send me the watermark text OR type **"skip"** to skip text watermark
5. Get your processed video with combined captions!

💡 **PNG Locations:** topleft, topright, bottomleft, bottomright (text watermark will skip PNG location and avoid center)
💡 **Skip Text Watermark:** Type "skip", "skip text", "no text", or "no watermark" to process video without text watermark but keep PNG watermark (if set).

Send me a video to start! 🎥
        """
        await message.reply_text(welcome_text)

    async def set_intro_handler(self, client: Client, message: Message):
        """Set intro command"""
        self.user_sessions[message.from_user.id] = {'waiting_for': 'set_intro'}
        await message.reply_text("🎬 Send me the intro video:")

    async def set_watermark_handler(self, client: Client, message: Message):
        """Set watermark command"""
        self.user_sessions[message.from_user.id] = {
            'waiting_for': 'set_watermark'
        }
        await message.reply_text(
            "🏷️ Send me the watermark image (PNG recommended):")

    async def set_thumbnail_handler(self, client: Client, message: Message):
        """Set thumbnail command"""
        self.user_sessions[message.from_user.id] = {
            'waiting_for': 'set_thumbnail'
        }
        await message.reply_text("🖼️ Send me the thumbnail image (JPG/PNG):")

    async def add_caption_handler(self, client: Client, message: Message):
        """Add permanent caption command"""
        self.user_sessions[message.from_user.id] = {
            'waiting_for': 'add_caption'
        }
        await message.reply_text("📝 Send me the permanent caption:")

    async def convert_handler(self, client: Client, message: Message):
        """Convert video with watermark, thumbnail, and caption"""
        user_id = message.from_user.id
        self.user_sessions[user_id] = {'waiting_for': 'convert_video'}
        await message.reply_text("🎥 Send me the MP4 video to convert:")

    async def status_handler(self, client: Client, message: Message):
        """Check status of saved items with file details"""
        user_id = message.from_user.id
        system_info = self.get_system_usage()

        items = {
            'Intro Video': f"persistent_intros/intro_{user_id}.mp4",
            'PNG Watermark': f"persistent_watermarks/watermark_{user_id}.png",
            'Thumbnail': f"persistent_thumbnails/thumbnail_{user_id}.jpg",
            'Permanent Caption': f"persistent_captions/caption_{user_id}.txt"
        }

        status_text = f"📊 **Your Settings Status:**\n{system_info}\n\n"

        for name, path in items.items():
            if os.path.exists(path):
                try:
                    file_size = os.path.getsize(path)
                    if file_size > 1024 * 1024:
                        size_str = f"{file_size/(1024*1024):.1f}MB"
                    elif file_size > 1024:
                        size_str = f"{file_size/1024:.1f}KB"
                    else:
                        size_str = f"{file_size}B"
                    status_text += f"✅ {name}: Set ({size_str})\n📁 {path}\n\n"
                except:
                    status_text += f"✅ {name}: Set\n📁 {path}\n\n"
            else:
                status_text += f"❌ {name}: Not set\n\n"

        # Check if directories exist
        status_text += "📂 **Directory Status:**\n"
        for directory in [
                'persistent_intros', 'persistent_watermarks',
                'persistent_thumbnails', 'persistent_captions'
        ]:
            exists = "✅" if os.path.exists(directory) else "❌"
            status_text += f"{exists} {directory}\n"

        await message.reply_text(status_text)

    async def stop_handler(self, client: Client, message: Message):
        """Stop current process with complete cleanup"""
        user_id = message.from_user.id
        session = self.user_sessions.get(user_id, {})

        # Mark as stopped FIRST to prevent new operations
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        self.user_sessions[user_id]['stopped'] = True

        # Kill all FFmpeg processes to stop persistent warnings
        try:
            import subprocess
            # Kill all FFmpeg processes
            subprocess.run(['pkill', '-f', 'ffmpeg'], check=False, capture_output=True)
            logger.info("Killed all FFmpeg processes")
        except Exception as e:
            logger.warning(f"Error killing FFmpeg processes: {e}")

        # Update progress message if exists
        if 'progress_message' in session:
            try:
                await session['progress_message'].edit_text(
                    "🛑 **Process stopped by user**")
                await asyncio.sleep(2)
                await session['progress_message'].delete()
            except:
                pass

        # Wait a moment for any ongoing operations to detect the stop flag
        await asyncio.sleep(1)

        # Clean up temp files with error handling
        user_dir = f"temp_{user_id}"
        if os.path.exists(user_dir):
            try:
                # Use a more robust cleanup that handles locked files
                for root, dirs, files in os.walk(user_dir, topdown=False):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except:
                            pass
                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            if os.path.exists(dir_path):
                                os.rmdir(dir_path)
                        except:
                            pass
                if os.path.exists(user_dir):
                    os.rmdir(user_dir)
                logger.info(f"Cleaned up temp directory: {user_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir: {e}")

        # Clean up normalized temp files in root directory
        try:
            import glob
            normalized_files = glob.glob("temp_*_normalized_*.mp4") + glob.glob("temp_list_*.txt")
            for temp_file in normalized_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.info(f"Cleaned up normalized temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {temp_file}: {e}")
        except Exception as e:
            logger.warning(f"Error cleaning up normalized files: {e}")

        # Clean up queue files
        queue_files = [f"bulk_queue/queue_{user_id}.json"]

        for queue_file in queue_files:
            if os.path.exists(queue_file):
                try:
                    os.remove(queue_file)
                    logger.info(f"Cleaned up queue file: {queue_file}")
                except Exception as e:
                    logger.warning(
                        f"Error cleaning up queue file {queue_file}: {e}")

        # Keep the stopped flag until next operation
        self.user_sessions[user_id] = {'stopped': True}

        await message.reply_text(
            "🛑 **All processes stopped and cleaned up!**\n✅ Temp files and queues cleared\n✅ FFmpeg processes terminated\n💡 Send /start to begin again"
        )

    async def bulk_handler(self, client: Client, message: Message):
        """Start bulk processing mode with enhanced messaging"""
        user_id = message.from_user.id
        watermark_path = f"persistent_watermarks/watermark_{user_id}.png"

        # Clear any existing queue
        queue_file = f"bulk_queue/queue_{user_id}.json"
        with open(queue_file, 'w') as f:
            json.dump([], f)

        if os.path.exists(watermark_path):
            self.user_sessions[user_id] = {
                'bulk_mode': True,
                'waiting_for': 'bulk_png_location',
                'bulk_queue_count': 0
            }
            await message.reply_text(
                "🔄 **Bulk Processing Mode Activated!**\n\n📍 **First, choose PNG watermark location:**\n• topleft\n• topright\n• bottomleft\n• bottomright\n\n✅ Send your choice, then add videos to queue!"
            )
        else:
            self.user_sessions[user_id] = {
                'bulk_mode': True,
                'bulk_queue_count': 0,
                'png_location': 'topright'  # Set default location
            }
            await message.reply_text(
                "🔄 **Bulk Processing Mode Activated!**\n\n📁 **Instructions:**\n1. Send multiple videos to add to queue\n2. Send watermark text to start processing\n\n💡 **Auto-start:** Type \"skip\" to process all videos without text watermark\n💡 **No size limits:** Send files of any size!"
            )

    # Media handlers
    async def video_handler(self, client: Client, message: Message):
        """Handle video uploads with metadata preservation"""
        user_id = message.from_user.id
        video = message.video
        session = self.user_sessions.get(user_id, {})

        # Initialize user session if it doesn't exist
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
            session = self.user_sessions[user_id]

        if session.get('stopped'):
            await message.reply_text(
                "⚠️ **Process was stopped**\nUse /start to begin again.")
            return

        if session.get('processing'):
            await message.reply_text(
                "⚠️ **Already processing a video**\nPlease wait or use /stop to cancel your current task."
            )
            return

        # Handle different modes
        if session.get('waiting_for') == 'set_intro':
            intro_path = f"persistent_intros/intro_{user_id}.mp4"
            await self.download_with_progress(message, video.file_id,
                                              intro_path, "intro")
            self.user_sessions[user_id] = {}
            await message.reply_text("✅ Intro video saved!")

        elif session.get('waiting_for') == 'convert_video':
            # Process video for conversion with watermark, thumbnail, and caption
            user_dir = f"temp_{user_id}"
            os.makedirs(user_dir, exist_ok=True)
            video_path = os.path.join(user_dir,
                                      f"convert_{int(time.time())}.mp4")

            await self.download_with_progress(message, video.file_id,
                                              video_path, "video to convert")

            # Watermark text prompt
            self.user_sessions[user_id] = {
                'video_path': video_path,
                'waiting_for': 'convert_watermark_text'
            }
            await message.reply_text(
                "✅ Video saved! Send the watermark text for conversion:\n\n💡 **Skip Text Watermark:** Type \"skip\" to convert video without text watermark but keep PNG watermark (if set)."
            )

        elif session.get('bulk_mode'):
            # Add to bulk queue with enhanced messaging
            queue_file = f"bulk_queue/queue_{user_id}.json"
            with open(queue_file, 'r') as f:
                queue = json.load(f)

            # Get filename with better processing
            caption = message.caption or f"video_{int(time.time())}"
            filename = getattr(video, 'file_name', None) or caption

            import re
            clean_filename = re.sub(r'[^\w\-_\.\s]', '_', filename)
            if not clean_filename.lower().endswith('.mp4'):
                clean_filename += '.mp4'

            # Get file size in MB for display
            file_size_mb = video.file_size / (1024 * 1024) if video.file_size else 0

            queue.append({
                'filename': clean_filename,
                'file_id': video.file_id,
                'file_size': video.file_size,
                'original_caption': caption,
                'duration': video.duration,
                'width': video.width,
                'height': video.height
            })

            with open(queue_file, 'w') as f:
                json.dump(queue, f)

            # Update session counter
            self.user_sessions[user_id]['bulk_queue_count'] = len(queue)

            # Enhanced queue message with file details
            queue_msg = f"✅ **Video #{len(queue)} added to queue!**\n"
            queue_msg += f"📂 **File:** {clean_filename}\n"
            if file_size_mb > 0:
                if file_size_mb >= 1024:
                    queue_msg += f"📊 **Size:** {file_size_mb/1024:.2f}GB\n"
                else:
                    queue_msg += f"📊 **Size:** {file_size_mb:.1f}MB\n"
            queue_msg += f"📋 **Total in queue:** {len(queue)} videos\n\n"

            # Auto-start suggestion for efficiency
            if len(queue) >= 3:
                queue_msg += "💡 **Ready to process?** Send watermark text or type \"skip\" to start processing automatically!"
            else:
                queue_msg += "📁 **Add more videos** or send watermark text to begin processing"

            await message.reply_text(queue_msg)

        else:
            # Regular single video processing - save metadata
            caption = message.caption or "input_video"
            import re
            clean_caption = re.sub(r'[^\w\-_\.]', '_', caption)
            if not clean_caption.endswith('.mp4'):
                clean_caption += '.mp4'

            # Store complete metadata in session
            self.user_sessions[user_id] = {
                'video_filename': clean_caption,
                'original_caption': caption,
                'video_file_id': video.file_id,
                'video_duration': video.duration or 0,
                'video_width': video.width or 1280,
                'video_height': video.height or 720,
                'waiting_for': 'png_location'
            }

            watermark_path = f"persistent_watermarks/watermark_{user_id}.png"
            if os.path.exists(watermark_path):
                await message.reply_text(
                    "✅ Video metadata saved! Choose PNG watermark location:\n\n📍 **Available locations:**\n• topleft\n• topright\n• bottomleft\n• bottomright\n\nSend your choice:"
                )
            else:
                self.user_sessions[user_id]['waiting_for'] = 'watermark_text'
                await message.reply_text(
                    "✅ Video metadata saved! Send me the watermark text:\n\n💡 **Skip Text Watermark:** Type \"skip\" to process video without any text watermark."
                )

    async def photo_handler(self, client: Client, message: Message):
        """Handle photo uploads"""
        user_id = message.from_user.id
        photo = message.photo
        session = self.user_sessions.get(user_id, {})

        if session.get('waiting_for') == 'set_watermark':
            await self.process_watermark_photo(message, user_id, photo)
        elif session.get('waiting_for') == 'set_thumbnail':
            await self.process_thumbnail_photo(message, user_id, photo)

    async def process_watermark_photo(self, message: Message, user_id: int,
                                      photo):
        """Process watermark photo"""
        temp_path = f"persistent_watermarks/temp_{user_id}.jpg"
        watermark_path = f"persistent_watermarks/watermark_{user_id}.png"

        # Ensure directory exists
        os.makedirs("persistent_watermarks", mode=0o755, exist_ok=True)

        await self.download_with_progress(message, photo.file_id, temp_path,
                                          "watermark image")

        try:
            with Image.open(temp_path) as img:
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                img.save(watermark_path, 'PNG', optimize=True)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Verify file was created
            if os.path.exists(watermark_path):
                file_size = os.path.getsize(watermark_path)
                logger.info(
                    f"Watermark saved: {watermark_path} ({file_size} bytes)")
                self.user_sessions[user_id] = {}
                await message.reply_text(
                    f"✅ PNG watermark saved successfully!\n📁 File: {watermark_path}"
                )
            else:
                raise Exception("Watermark file was not created")

        except Exception as e:
            logger.error(f"Watermark processing error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await message.reply_text(f"❌ Error processing watermark: {str(e)}")

    async def document_handler(self, client: Client, message: Message):
        """Handle document uploads (for PNG watermarks)"""
        user_id = message.from_user.id
        document = message.document
        session = self.user_sessions.get(user_id, {})

        if session.get('waiting_for') == 'set_watermark':
            if document.mime_type in ['image/png', 'image/jpeg', 'image/jpg'
                                      ] or document.file_name.lower().endswith(
                                          ('.png', '.jpg', '.jpeg')):
                await self.process_watermark_document(message, user_id,
                                                      document)
            else:
                await message.reply_text(
                    "❌ Please send a valid PNG, JPG, or JPEG image file.")
        elif session.get('waiting_for') == 'set_thumbnail':
            if document.mime_type in ['image/png', 'image/jpeg', 'image/jpg'
                                      ] or document.file_name.lower().endswith(
                                          ('.png', '.jpg', '.jpeg')):
                await self.process_thumbnail_document(message, user_id,
                                                      document)
            else:
                await message.reply_text(
                    "❌ Please send a valid PNG, JPG, or JPEG image file.")

    async def process_watermark_document(self, message: Message, user_id: int,
                                         document):
        """Process document watermark"""
        watermark_path = f"persistent_watermarks/watermark_{user_id}.png"
        await self.download_with_progress(message, document.file_id,
                                          watermark_path, "watermark document")

        try:
            with Image.open(watermark_path) as img:
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                img.save(watermark_path, 'PNG', optimize=True)

            self.user_sessions[user_id] = {}
            await message.reply_text("✅ PNG watermark saved successfully!")

        except Exception as e:
            logger.error(f"Document watermark error: {e}")
            if os.path.exists(watermark_path):
                os.remove(watermark_path)
            await message.reply_text(f"❌ Error processing watermark: {str(e)}")

    async def process_thumbnail_photo(self, message: Message, user_id: int,
                                      photo):
        """Process thumbnail photo"""
        temp_path = f"persistent_thumbnails/temp_{user_id}.jpg"
        thumbnail_path = f"persistent_thumbnails/thumbnail_{user_id}.jpg"

        # Ensure directory exists
        os.makedirs("persistent_thumbnails", mode=0o755, exist_ok=True)

        await self.download_with_progress(message, photo.file_id, temp_path,
                                          "thumbnail image")

        try:
            with Image.open(temp_path) as img:
                # Convert to RGB if needed (for JPG compatibility)
                if img.mode in ('RGBA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        rgb_img.paste(img, mask=img.split()[-1])
                    else:
                        rgb_img.paste(img)
                    img = rgb_img

                # Resize to reasonable thumbnail size (320x180 for 16:9)
                img.thumbnail((320, 180), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Verify file was created
            if os.path.exists(thumbnail_path):
                file_size = os.path.getsize(thumbnail_path)
                logger.info(
                    f"Thumbnail saved: {thumbnail_path} ({file_size} bytes)")
                self.user_sessions[user_id] = {}
                await message.reply_text(
                    f"✅ Thumbnail saved successfully!\n📁 File: {thumbnail_path}"
                )
            else:
                raise Exception("Thumbnail file was not created")

        except Exception as e:
            logger.error(f"Thumbnail processing error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await message.reply_text(f"❌ Error processing thumbnail: {str(e)}")

    async def process_thumbnail_document(self, message: Message, user_id: int,
                                         document):
        """Process document thumbnail"""
        thumbnail_path = f"persistent_thumbnails/thumbnail_{user_id}.jpg"
        await self.download_with_progress(message, document.file_id,
                                          thumbnail_path, "thumbnail document")

        try:
            with Image.open(thumbnail_path) as img:
                # Convert to RGB if needed (for JPG compatibility)
                if img.mode in ('RGBA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        rgb_img.paste(img, mask=img.split()[-1])
                    else:
                        rgb_img.paste(img)
                    img = rgb_img

                # Resize to reasonable thumbnail size (320x180 for 16:9)
                img.thumbnail((320, 180), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)

            self.user_sessions[user_id] = {}
            await message.reply_text("✅ Thumbnail saved successfully!")

        except Exception as e:
            logger.error(f"Document thumbnail error: {e}")
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
            await message.reply_text(f"❌ Error processing thumbnail: {str(e)}")

    async def text_handler(self, client: Client, message: Message):
        """Handle text messages"""
        user_id = message.from_user.id
        text = message.text
        session = self.user_sessions.get(user_id, {})

        # Initialize user session if it doesn't exist
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
            session = self.user_sessions[user_id]

        if session.get('stopped'):
            await message.reply_text(
                "⚠️ **Process was stopped**\nUse /start to begin again.")
            return

        if session.get('processing'):
            await message.reply_text(
                "⚠️ **Already processing a video**\nPlease wait or use /stop to cancel your current task."
            )
            return

        # Clear stopped flag when starting new operations
        if session.get('stopped'):
            self.user_sessions[user_id].pop('stopped', None)

        # Debug: Log current session state for troubleshooting
        current_state = session.get('waiting_for', 'none')
        logger.info(f"User {user_id} text handler - Session state: {current_state}, Text: {text[:50]}...")

        # Handle permanent caption setting
        if session.get('waiting_for') == 'add_caption':
            await self.process_add_caption(message, user_id, text)

        # Handle PNG location selection
        elif session.get('waiting_for') == 'png_location':
            valid_locations = [
                'topleft', 'topright', 'bottomleft', 'bottomright'
            ]
            location = text.lower().strip()

            if location in valid_locations:
                self.user_sessions[user_id]['png_location'] = location
                self.user_sessions[user_id]['waiting_for'] = 'watermark_text'
                await message.reply_text(
                    f"✅ PNG location set to {location}! Now send watermark text:\n\n💡 **Skip Text Watermark:** Type \"skip\" to process video without text watermark but keep PNG watermark."
                )
            else:
                await message.reply_text(
                    f"❌ Invalid location. Please choose from: {', '.join(valid_locations)}"
                )

        # Handle bulk PNG location selection
        elif session.get('waiting_for') == 'bulk_png_location':
            valid_locations = [
                'topleft', 'topright', 'bottomleft', 'bottomright'
            ]
            location = text.lower().strip()

            if location in valid_locations:
                self.user_sessions[user_id]['png_location'] = location
                self.user_sessions[user_id].pop('waiting_for', None)
                await message.reply_text(
                    f"✅ PNG location set to {location}! Now send multiple videos, then watermark text:\n\n💡 **Skip Text Watermark:** Type \"skip\" to process all videos without text watermark but keep PNG watermark."
                )
            else:
                await message.reply_text(
                    f"❌ Invalid location. Please choose from: {', '.join(valid_locations)}"
                )

        # Handle bulk processing
        elif session.get('bulk_mode'):
            # Check if we're still waiting for PNG location
            if session.get('waiting_for') == 'bulk_png_location':
                # This is handled above, so skip here
                return
            
            # Check for skip command
            if text.lower() in [
                    'skip', 'skip text', 'no text', 'no watermark'
            ]:
                png_location = session.get('png_location', 'topright')
                await self.process_bulk_queue(
                    message, user_id, None,
                    png_location)  # Pass None for no text watermark
            else:
                png_location = session.get('png_location', 'topright')
                await self.process_bulk_queue(message, user_id, text,
                                              png_location)

        # Handle convert video watermark text
        elif session.get('waiting_for') == 'convert_watermark_text':
            session = self.user_sessions[user_id]
            video_path = session.get('video_path')
            if video_path:
                # Check for skip command
                if text.lower() in [
                        'skip', 'skip text', 'no text', 'no watermark'
                ]:
                    watermark_text = None
                else:
                    watermark_text = text

                self.user_sessions[user_id] = {
                    'video_path': video_path,
                    'watermark_text': watermark_text,
                    'waiting_for': 'process_convert_video'
                }
                skip_msg = " (skipping text watermark)" if watermark_text is None else ""
                await message.reply_text(
                    f"✅ Watermark text saved{skip_msg}! Processing video for conversion..."
                )
                await self.process_convert_video(message, user_id, video_path,
                                                 watermark_text)
            else:
                await message.reply_text(
                    "❌ No video found for conversion. Please send a video using /convert first."
                )

        # Handle single video watermarking
        elif session.get('waiting_for') == 'watermark_text':
            # Check for skip command
            if text.lower() in [
                    'skip', 'skip text', 'no text', 'no watermark'
            ]:
                watermark_text = None
            else:
                watermark_text = text

            # Mark this specific user as processing
            self.user_sessions[user_id]['processing'] = True
            try:
                user_dir = f"temp_{user_id}"
                os.makedirs(user_dir, exist_ok=True)
                video_path = os.path.join(user_dir, session['video_filename'])
                png_location = session.get('png_location',
                                           'topright')  # Default to topright

                await self.process_video_with_metadata(
                    message,
                    user_id,
                    video_path,
                    watermark_text,
                    session['original_caption'],
                    png_location=png_location)
            except Exception as e:
                logger.error(f"Error in video processing for user {user_id}: {e}")
                await message.reply_text(f"❌ Error processing video: {str(e)}")
            finally:
                # Always clean up the processing flag for this user
                if user_id in self.user_sessions and 'processing' in self.user_sessions[user_id]:
                    self.user_sessions[user_id].pop('processing', None)

        else:
            # Check if user might be trying to send watermark text without proper session state
            if text.strip() and not text.startswith('/'):
                await message.reply_text(
                    "Please send a video first, then I'll ask for watermark text.\n\nUse /start for help or send a video to begin watermarking."
                )
            else:
                await message.reply_text(
                    "Please send a video first or use /start for help."
                )

    async def process_add_caption(self, message: Message, user_id: int,
                                  caption_text: str):
        """Process add caption command"""
        caption_path = f"persistent_captions/caption_{user_id}.txt"
        try:
            # Ensure directory exists
            os.makedirs("persistent_captions", mode=0o755, exist_ok=True)

            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption_text)

            # Verify file was created
            if os.path.exists(caption_path):
                file_size = os.path.getsize(caption_path)
                logger.info(
                    f"Caption file saved: {caption_path} ({file_size} bytes)")
                self.user_sessions[user_id] = {}
                await message.reply_text(
                    f"✅ Permanent caption saved successfully!\n📁 File: {caption_path}"
                )
            else:
                raise Exception("Caption file was not created")

        except Exception as e:
            logger.error(f"Caption saving error: {e}")
            await message.reply_text(f"❌ Error saving caption: {str(e)}")

    async def process_convert_video(self, message: Message, user_id: int,
                                    video_path: str, watermark_text: str):
        """Process the convert video request"""
        try:
            session = self.user_sessions.get(user_id, {})
            if session.get('stopped'):
                await message.reply_text("🛑 **Process stopped by user**")
                return

            # Set output file path
            converted_path = video_path.replace(".mp4", "_converted.mp4")

            # Get thumbnail path
            thumbnail_path = f"persistent_thumbnails/thumbnail_{user_id}.jpg"
            thumbnail = thumbnail_path if os.path.exists(
                thumbnail_path) else None

            # Get caption
            permanent_caption_path = f"persistent_captions/caption_{user_id}.txt"
            permanent_caption = ""
            if os.path.exists(permanent_caption_path):
                try:
                    with open(permanent_caption_path, 'r') as f:
                        permanent_caption = f.read().strip()
                except Exception as e:
                    logger.error(f"Error reading permanent caption: {e}")

            # Add combined watermarks, thumbnail, and caption
            await self.add_watermark_thumbnail_caption(
                message, user_id, video_path, converted_path, watermark_text,
                permanent_caption, thumbnail)

        except Exception as e:
            logger.error(f"Video conversion error: {e}")
            await message.reply_text(f"❌ Error converting video: {str(e)}")
        finally:
            self.user_sessions[user_id] = {}  # Reset user session

    async def add_watermark_thumbnail_caption(self,
                                              message: Message,
                                              user_id: int,
                                              input_path: str,
                                              output_path: str,
                                              watermark_text: str,
                                              caption_text: str,
                                              thumbnail: str = None):
        """Add watermark, thumbnail, and caption to video"""
        import subprocess

        try:
            # Get video info
            video_info = await self.get_video_info(input_path)
            width = video_info['width']
            height = video_info['height']
            fps = video_info['fps']

            if fps <= 0 or fps > 120:
                fps = 30

            # Build filter complex
            filter_parts = []
            overlay_inputs = "[0:v]"

            # Add text watermark if provided
            if watermark_text:
                # Clean and escape the text for FFmpeg, but preserve actual line breaks
                clean_text = watermark_text.replace("'", "\\'").replace(":", "\\:")
                # Remove any characters that might break FFmpeg filters but keep newlines
                clean_text = ''.join(
                    c for c in clean_text
                    if ord(c) < 127 and (c.isprintable() or c in ['\n', '\r', ' ']))

                # Split by actual line breaks first (user's intended lines)
                user_lines = clean_text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

                # Process each user line for length and create final lines
                lines = []
                max_chars_per_line = max(25, width // 12)  # Slightly longer lines for better readability

                for user_line in user_lines:
                    user_line = user_line.strip()
                    if not user_line and len(lines) == 0:  # Skip empty lines only at the beginning
                        continue
                    elif not user_line:  # Preserve empty lines in the middle/end as spacing
                        lines.append(" ")  # Use space instead of empty to maintain line structure
                        continue

                    # If line is short enough, use as-is
                    if len(user_line) <= max_chars_per_line:
                        lines.append(user_line)
                    else:
                        # Split long lines by words
                        words = user_line.split()
                        current_line = ""

                        for word in words:
                            test_line = current_line + " " + word if current_line else word
                            if len(test_line) <= max_chars_per_line:
                                current_line = test_line
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word

                        if current_line:
                            lines.append(current_line)

                # Limit to 5 lines maximum for better display (increased from 4)
                if len(lines) > 5:
                    lines = lines[:4]
                    lines.append("...")

                # Join lines with proper newline character for FFmpeg
                # Use \n for proper line breaks in FFmpeg drawtext (not \\n)
                final_text = "\\n".join(lines)

                font_size = max(18, min(width // 40, height //
                                        30))  # Slightly smaller for multi-line
                margin = 15
                line_spacing = int(font_size * 1.2)  # Line spacing

                # Only 3 positions: topright -> bottomright -> bottomleft (never use topleft)
                # 3 positions: 3 * 60 = 180 seconds cycle
                text_filter = (
                    f"drawtext=text='{final_text}':"
                    f"fontsize={font_size}:"
                    f"fontcolor=white@0.95:"
                    f"box=1:boxcolor=black@0.8:boxborderw=10:"
                    f"line_spacing={line_spacing}:"
                    f"x='if(lt(mod(t,180),60),w-text_w-{margin},if(lt(mod(t,180),120),w-text_w-{margin},{margin}))':"
                    f"y='if(lt(mod(t,180),60),{margin*2},if(lt(mod(t,180),120),h-text_h-{margin},h-text_h-{margin}))'"
                )

                filter_parts.append(f"{overlay_inputs}{text_filter}[txt]")
                overlay_inputs = "[txt]"

            # Build command optimized for same file size as original
            cmd = [
                'ffmpeg',
                '-i',
                input_path,
                '-vf',
                ';'.join(filter_parts) if filter_parts else 'copy',
                '-c:v',
                'libx264',
                '-preset',
                'ultrafast',
                '-crf',
                '23',  # Faster encoding, reasonable quality
                '-c:a',
                'copy',  # Copy audio to maintain original quality and speed
                '-r',
                str(fps),
                '-profile:v',
                'main',
                '-level',
                '3.1',  # Standard profile for compatibility
                '-map_metadata',
                '0',  # Preserve metadata
                '-movflags',
                '+faststart',
                '-threads',
                '0',  # Use all available CPU threads
                '-y',
                output_path
            ]

            # Execute command
            process = subprocess.Popen(cmd,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True)
            process.wait()

            if process.returncode == 0:
                logger.info("Successfully added watermark and thumbnail")

                # Send the video with caption and thumbnail
                try:
                    await self.app.send_video(chat_id=user_id,
                                              video=output_path,
                                              caption=caption_text,
                                              thumb=thumbnail,
                                              supports_streaming=True)
                except Exception as send_error:
                    logger.error(
                        f"Error sending converted video: {send_error}")
                    await message.reply_text(
                        f"❌ Error sending converted video: {send_error}")

            else:
                stderr_output = process.stderr.read(
                ) if process.stderr else "Unknown error"
                logger.error(f"Conversion failed with stderr: {stderr_output}")
                await message.reply_text(
                    f"❌ Conversion failed: {stderr_output}")

        except Exception as e:
            logger.error(f"Conversion process error: {e}")
            await message.reply_text(f"❌ Conversion process error: {str(e)}")

    async def remove_intro_handler(self, client: Client, message: Message):
        await self.remove_persistent_item(message, 'intro')

    async def remove_watermark_handler(self, client: Client, message: Message):
        await self.remove_persistent_item(message, 'watermark')

    async def remove_thumbnail_handler(self, client: Client, message: Message):
        await self.remove_persistent_item(message, 'thumbnail')

    async def remove_caption_handler(self, client: Client, message: Message):
        await self.remove_persistent_item(message, 'caption')

    async def queue_handler(self, client: Client, message: Message):
        """Check bulk processing queue status"""
        user_id = message.from_user.id
        queue_file = f"bulk_queue/queue_{user_id}.json"

        status_text = "📊 **Queue Status:**\n\n"

        # Check main queue
        if os.path.exists(queue_file):
            with open(queue_file, 'r') as f:
                queue = json.load(f)
            status_text += f"🎬 Main Queue: {len(queue)} videos\n"
        else:
            status_text += "🎬 Main Queue: Empty\n"

        await message.reply_text(status_text)

    async def adduser_handler(self, client: Client, message: Message):
        """Add user to admin list (admin only)"""
        user_id = message.from_user.id
        if user_id not in self.admin_users:
            await message.reply_text(
                "❌ You don't have permission to use this command.")
            return

        try:
            command_parts = message.text.split()
            if len(command_parts) < 2:
                await message.reply_text("❌ Usage: /adduser <user_id>")
                return

            new_admin_id = int(command_parts[1])
            if new_admin_id in self.admin_users:
                await message.reply_text(
                    f"ℹ️ User {new_admin_id} is already an admin.")
            else:
                self.admin_users.append(new_admin_id)
                await message.reply_text(
                    f"✅ User {new_admin_id} added to admin list.")
                logger.info(
                    f"Admin {user_id} added user {new_admin_id} to admin list")
        except ValueError:
            await message.reply_text(
                "❌ Invalid user ID. Please provide a valid number.")
        except Exception as e:
            await message.reply_text(f"❌ Error adding user: {str(e)}")

    async def removeuser_handler(self, client: Client, message: Message):
        """Remove user from admin list (admin only)"""
        user_id = message.from_user.id
        if user_id not in self.admin_users:
            await message.reply_text(
                "❌ You don't have permission to use this command.")
            return

        try:
            command_parts = message.text.split()
            if len(command_parts) < 2:
                await message.reply_text("❌ Usage: /removeuser <user_id>")
                return

            remove_admin_id = int(command_parts[1])
            if remove_admin_id == 2038923790:  # Protect main admin
                await message.reply_text("❌ Cannot remove the main admin.")
                return

            if remove_admin_id in self.admin_users:
                self.admin_users.remove(remove_admin_id)
                await message.reply_text(
                    f"✅ User {remove_admin_id} removed from admin list.")
                logger.info(
                    f"Admin {user_id} removed user {remove_admin_id} from admin list"
                )
            else:
                await message.reply_text(
                    f"ℹ️ User {remove_admin_id} is not an admin.")
        except ValueError:
            await message.reply_text(
                "❌ Invalid user ID. Please provide a valid number.")
        except Exception as e:
            await message.reply_text(f"❌ Error removing user: {str(e)}")

    async def listadmins_handler(self, client: Client, message: Message):
        """List all admin users (admin only)"""
        user_id = message.from_user.id
        if user_id not in self.admin_users:
            await message.reply_text(
                "❌ You don't have permission to use this command.")
            return

        admin_list = "🔧 **Admin Users:**\n\n"
        for i, admin_id in enumerate(self.admin_users, 1):
            admin_list += f"{i}. {admin_id}"
            if admin_id == 2038923790:
                admin_list += " (Main Admin)"
            admin_list += "\n"

        await message.reply_text(admin_list)

    async def stats_handler(self, client: Client, message: Message):
        """Get detailed bot statistics (admin only)"""
        user_id = message.from_user.id
        if user_id not in self.admin_users:
            await message.reply_text(
                "❌ You don't have permission to use this command.")
            return

        system_info = self.get_system_usage()

        # Count files in directories
        stats_text = f"📊 **Bot Statistics:**\n{system_info}\n\n"

        directories = {
            'Intros': 'persistent_intros',
            'Watermarks': 'persistent_watermarks',
            'Thumbnails': 'persistent_thumbnails',
            'Captions': 'persistent_captions',
            'Bulk Queues': 'bulk_queue',
            'Normalized Cache': 'normalized_intros_cache'
        }

        for name, path in directories.items():
            if os.path.exists(path):
                file_count = len([
                    f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                ])
                stats_text += f"📁 {name}: {file_count} files\n"
            else:
                stats_text += f"📁 {name}: Directory not found\n"

        # Active sessions
        active_sessions = len([s for s in self.user_sessions.values() if s])
        stats_text += f"👥 Active Sessions: {active_sessions}\n"
        stats_text += f"🔧 Admin Users: {len(self.admin_users)}\n"

        await message.reply_text(stats_text)

    async def cleanup_handler(self, client: Client, message: Message):
        """Clean all temp and cache files (admin only)"""
        user_id = message.from_user.id
        if user_id not in self.admin_users:
            await message.reply_text(
                "❌ You don't have permission to use this command.")
            return

        try:
            cleaned_count = 0

            # Clean temp directories
            for root, dirs, files in os.walk('.'):
                for dir_name in dirs:
                    if dir_name.startswith('temp_'):
                        temp_dir = os.path.join(root, dir_name)
                        try:
                            shutil.rmtree(temp_dir)
                            cleaned_count += 1
                        except:
                            pass

            # Clean cache
            cache_dir = "normalized_intros_cache"
            if os.path.exists(cache_dir):
                for cache_file in os.listdir(cache_dir):
                    try:
                        os.remove(os.path.join(cache_dir, cache_file))
                        cleaned_count += 1
                    except:
                        pass

            # Clean any temp files in root
            for file in os.listdir('.'):
                if file.startswith('temp_') and os.path.isfile(file):
                    try:
                        os.remove(file)
                        cleaned_count += 1
                    except:
                        pass

            await message.reply_text(
                f"✅ Cleanup completed! Removed {cleaned_count} items.")
            logger.info(
                f"Admin {user_id} performed cleanup, removed {cleaned_count} items"
            )

        except Exception as e:
            await message.reply_text(f"❌ Cleanup error: {str(e)}")
            logger.error(f"Cleanup error: {e}")

    async def process_bulk_queue(self, message: Message, user_id: int, watermark_text: str, png_location: str):
        """Process all videos in the bulk queue"""
        queue_file = f"bulk_queue/queue_{user_id}.json"
        
        try:
            if not os.path.exists(queue_file):
                await message.reply_text("❌ No videos in queue. Add videos first.")
                return

            with open(queue_file, 'r') as f:
                queue = json.load(f)

            if not queue:
                await message.reply_text("❌ Queue is empty. Add videos first.")
                return

            # Mark this specific user as processing and clear bulk mode
            self.user_sessions[user_id]['processing'] = True
            self.user_sessions[user_id]['bulk_mode'] = False
            self.user_sessions[user_id]['total_videos'] = len(queue)

            skip_msg = " (skipping text watermark)" if watermark_text is None else ""
            await message.reply_text(
                f"🚀 **Starting bulk processing of {len(queue)} videos{skip_msg}!**\n⏳ Processing will begin shortly..."
            )

            # Process each video in the queue
            for i, video_info in enumerate(queue, 1):
                try:
                    # Check if stopped
                    if self.user_sessions.get(user_id, {}).get('stopped'):
                        await message.reply_text("🛑 **Bulk processing stopped by user**")
                        break

                    # Create temp directory for this video
                    user_dir = f"temp_{user_id}"
                    os.makedirs(user_dir, exist_ok=True)
                    video_path = os.path.join(user_dir, video_info['filename'])

                    # Store video metadata in session for processing
                    self.user_sessions[user_id].update({
                        'video_file_id': video_info['file_id'],
                        'video_duration': video_info.get('duration', 0),
                        'video_width': video_info.get('width', 1280),
                        'video_height': video_info.get('height', 720)
                    })

                    # Process this video
                    await self.process_video_with_metadata(
                        message,
                        user_id,
                        video_path,
                        watermark_text,
                        video_info['original_caption'],
                        video_num=i,
                        png_location=png_location
                    )

                    # Small delay between videos
                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"Error processing video {i}: {e}")
                    await message.reply_text(f"❌ Error processing video {i}: {str(e)}")
                    continue

            # Final completion message
            await message.reply_text(
                f"🎉 **Bulk processing completed!**\n✅ Processed {len(queue)} videos successfully"
            )

        except Exception as e:
            logger.error(f"Bulk processing error: {e}")
            await message.reply_text(f"❌ Bulk processing error: {str(e)}")
        finally:
            # Reset session
            if user_id in self.user_sessions:
                self.user_sessions[user_id] = {}

            # Clear queue file
            if os.path.exists(queue_file):
                try:
                    os.remove(queue_file)
                except:
                    pass

    async def remove_persistent_item(self, message: Message, item_type: str):
        """Remove persistent item and related cache"""
        user_id = message.from_user.id
        extensions = {
            'intro': 'mp4',
            'watermark': 'png',
            'thumbnail': 'jpg',
            'caption': 'txt'
        }
        file_path = f"persistent_{item_type}s/{item_type}_{user_id}.{extensions[item_type]}"

        if os.path.exists(file_path):
            os.remove(file_path)

            # If removing intro, also clean cache
            if item_type == 'intro':
                cache_dir = "normalized_intros_cache"
                if os.path.exists(cache_dir):
                    try:
                        for cache_file in os.listdir(cache_dir):
                            if f"intro_" in cache_file:  # Clean all cached intros for safety
                                cache_path = os.path.join(
                                    cache_dir, cache_file)
                                if os.path.exists(cache_path):
                                    os.remove(cache_path)
                    except Exception as e:
                        logger.warning(f"Cache cleanup warning: {e}")

            await message.reply_text(
                f"✅ {item_type.title()} removed successfully! (Cache cleared)")
        else:
            await message.reply_text(f"❌ No {item_type} found to remove.")

    def run(self):
        """Run the bot"""
        from pyrogram import handlers

        # Add all handlers
        self.app.add_handler(
            handlers.MessageHandler(self.start_handler,
                                    pyrogram_filters.command("start")))
        self.app.add_handler(
            handlers.MessageHandler(self.set_intro_handler,
                                    pyrogram_filters.command("setintro")))
        self.app.add_handler(
            handlers.MessageHandler(self.set_watermark_handler,
                                    pyrogram_filters.command("setwatermark")))
        self.app.add_handler(
            handlers.MessageHandler(self.set_thumbnail_handler,
                                    pyrogram_filters.command("setthumbnail")))
        self.app.add_handler(
            handlers.MessageHandler(self.add_caption_handler,
                                    pyrogram_filters.command("addcaption")))
        self.app.add_handler(
            handlers.MessageHandler(self.convert_handler,
                                    pyrogram_filters.command("convert")))
        self.app.add_handler(
            handlers.MessageHandler(self.status_handler,
                                    pyrogram_filters.command("status")))
        self.app.add_handler(
            handlers.MessageHandler(self.stop_handler,
                                    pyrogram_filters.command("stop")))
        self.app.add_handler(
            handlers.MessageHandler(self.bulk_handler,
                                    pyrogram_filters.command("bulk")))
        self.app.add_handler(
            handlers.MessageHandler(self.remove_intro_handler,
                                    pyrogram_filters.command("removeintro")))
        self.app.add_handler(
            handlers.MessageHandler(
                self.remove_watermark_handler,
                pyrogram_filters.command("removewatermark")))
        self.app.add_handler(
            handlers.MessageHandler(
                self.remove_thumbnail_handler,
                pyrogram_filters.command("removethumbnail")))
        self.app.add_handler(
            handlers.MessageHandler(self.remove_caption_handler,
                                    pyrogram_filters.command("removecaption")))
        self.app.add_handler(
            handlers.MessageHandler(self.queue_handler,
                                    pyrogram_filters.command("queue")))
        self.app.add_handler(
            handlers.MessageHandler(self.adduser_handler,
                                    pyrogram_filters.command("adduser")))
        self.app.add_handler(
            handlers.MessageHandler(self.removeuser_handler,
                                    pyrogram_filters.command("removeuser")))
        self.app.add_handler(
            handlers.MessageHandler(self.listadmins_handler,
                                    pyrogram_filters.command("listadmins")))
        self.app.add_handler(
            handlers.MessageHandler(self.stats_handler,
                                    pyrogram_filters.command("stats")))
        self.app.add_handler(
            handlers.MessageHandler(self.cleanup_handler,
                                    pyrogram_filters.command("cleanup")))

        # Media handlers
        self.app.add_handler(
            handlers.MessageHandler(self.video_handler,
                                    pyrogram_filters.video))
        self.app.add_handler(
            handlers.MessageHandler(self.photo_handler,
                                    pyrogram_filters.photo))
        self.app.add_handler(
            handlers.MessageHandler(self.document_handler,
                                    pyrogram_filters.document))
        self.app.add_handler(
            handlers.MessageHandler(
                self.text_handler,
                pyrogram_filters.text & ~pyrogram_filters.command([
                    "start", "setintro", "setwatermark", "setthumbnail",
                    "addcaption", "convert", "status", "stop", "bulk",
                    "removeintro", "removewatermark", "removethumbnail",
                    "removecaption", "queue", "adduser", "removeuser",
                    "listadmins", "stats", "cleanup"
                ])))

        print("🤖 Watermark Bot starting...")

        try:
            self.app.run()
        except Exception as e:
            logger.error(f"Bot error: {e}")
            raise


if __name__ == "__main__":
    API_ID = int(os.getenv("API_ID", "6063221"))
    API_HASH = os.getenv("API_HASH", "8f9bebe9a9cb147ee58f70f46506f787")
    BOT_TOKEN = os.getenv("BOT_TOKEN",
                          "7219961311:AAHllFimFwhoMRIJSJy1OSQ_oviiGceK-m4")

    print("🤖 Starting enhanced watermark bot...")
    bot = WatermarkBot(API_ID, API_HASH, BOT_TOKEN)
    bot.run()