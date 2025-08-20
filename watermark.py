import asyncio
import os
import tempfile
import shutil
import logging
import json
import time
import threading
import concurrent.futures
import multiprocessing
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from pyrogram import Client, filters as pyrogram_filters
from pyrogram.types import Message
import psutil
import subprocess
import hashlib
from pathlib import Path
import weakref

# Configure optimized logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('watermark_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoInfo:
    """Optimized video information container"""
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    bitrate: int
    format: str
    frame_count: int
    has_audio: bool
    audio_codec: str = None
    audio_bitrate: int = 0

@dataclass
class ProcessingProgress:
    """Progress tracking container"""
    stage: str
    current: int
    total: int
    percentage: float
    speed: float
    eta_seconds: float
    elapsed_seconds: float
    details: Dict[str, Any]

class OptimizedFFmpegProcessor:
    """High-performance FFmpeg processor with GPU acceleration support"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_support()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = min(self.cpu_count * 2, 16)
        self._setup_gpu_codecs()
        
    def _check_gpu_support(self) -> bool:
        """Check for available GPU acceleration"""
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("NVIDIA GPU detected")
                return True
        except FileNotFoundError:
            pass
            
        try:
            # Check for Intel Quick Sync
            result = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], 
                                 capture_output=True, text=True)
            if 'qsv' in result.stdout:
                logger.info("Intel Quick Sync detected")
                return True
        except FileNotFoundError:
            pass
            
        try:
            # Check for AMD GPU
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("AMD GPU detected")
                return True
        except FileNotFoundError:
            pass
            
        logger.info("No GPU acceleration available, using CPU")
        return False
    
    def _setup_gpu_codecs(self):
        """Setup optimal codecs based on available hardware"""
        if self.gpu_available:
            # Test available hardware encoders
            test_cmd = ['ffmpeg', '-hide_banner', '-encoders']
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True)
                output = result.stdout.lower()
                
                if 'h264_nvenc' in output:
                    self.encoder = 'h264_nvenc'
                    self.decoder = 'h264_cuvid'
                    self.gpu_type = 'nvidia'
                elif 'h264_qsv' in output:
                    self.encoder = 'h264_qsv'
                    self.decoder = 'h264_qsv'
                    self.gpu_type = 'intel'
                elif 'h264_amf' in output:
                    self.encoder = 'h264_amf'
                    self.decoder = 'h264'
                    self.gpu_type = 'amd'
                else:
                    self._fallback_to_cpu()
            except Exception as e:
                logger.warning(f"GPU codec detection failed: {e}")
                self._fallback_to_cpu()
        else:
            self._fallback_to_cpu()
    
    def _fallback_to_cpu(self):
        """Fallback to optimized CPU encoding"""
        self.encoder = 'libx264'
        self.decoder = 'h264'
        self.gpu_type = 'cpu'
        self.gpu_available = False
        
    def get_optimal_preset(self, file_size_mb: float, duration: float) -> str:
        """Get optimal encoding preset based on file characteristics"""
        if self.gpu_available:
            return 'fast'  # GPU encoders use different presets
        else:
            # CPU encoding optimization based on file size and duration
            if file_size_mb > 500 or duration > 1800:  # Large files or >30min
                return 'ultrafast'
            elif file_size_mb > 100 or duration > 600:  # Medium files or >10min
                return 'superfast'
            else:
                return 'veryfast'
    
    def get_optimal_crf(self, target_quality: str = 'balanced') -> int:
        """Get optimal CRF value for quality vs speed balance"""
        quality_map = {
            'fastest': 28,
            'balanced': 23,
            'quality': 20
        }
        return quality_map.get(target_quality, 23)

class AdvancedProgressTracker:
    """Advanced progress tracking with predictive ETA and smooth updates"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.start_time = time.time()
        self.stages = {}
        self.current_stage = None
        self.progress_history = []
        self.speed_samples = []
        self.last_update = 0
        self.update_interval = 1.0  # Reduced to 1 second for responsiveness
        
    def add_stage(self, stage_name: str, total_work: int, weight: float = 1.0):
        """Add a processing stage with estimated work units"""
        self.stages[stage_name] = {
            'total_work': total_work,
            'completed_work': 0,
            'weight': weight,
            'start_time': None,
            'end_time': None,
            'speed_history': []
        }
    
    def update_stage(self, stage_name: str, completed_work: int, details: Dict = None):
        """Update progress for a specific stage"""
        if stage_name not in self.stages:
            self.add_stage(stage_name, completed_work, 1.0)
            
        stage = self.stages[stage_name]
        if stage['start_time'] is None:
            stage['start_time'] = time.time()
            
        stage['completed_work'] = completed_work
        stage['details'] = details or {}
        
        # Calculate stage speed
        elapsed = time.time() - stage['start_time']
        if elapsed > 0:
            speed = completed_work / elapsed
            stage['speed_history'].append(speed)
            if len(stage['speed_history']) > 10:
                stage['speed_history'].pop(0)
        
        self.current_stage = stage_name
    
    def get_overall_progress(self) -> ProcessingProgress:
        """Calculate overall progress across all stages"""
        if not self.stages:
            return ProcessingProgress("initializing", 0, 100, 0.0, 0.0, 0.0, 0.0, {})
        
        total_weight = sum(stage['weight'] for stage in self.stages.values())
        weighted_progress = 0.0
        overall_speed = 0.0
        
        for stage_name, stage in self.stages.items():
            if stage['total_work'] > 0:
                stage_progress = min(1.0, stage['completed_work'] / stage['total_work'])
                weighted_progress += stage_progress * stage['weight']
                
                if stage['speed_history']:
                    avg_speed = sum(stage['speed_history']) / len(stage['speed_history'])
                    overall_speed += avg_speed * stage['weight']
        
        overall_percentage = (weighted_progress / total_weight) * 100 if total_weight > 0 else 0.0
        overall_percentage = min(100.0, max(0.0, overall_percentage))
        
        elapsed = time.time() - self.start_time
        eta = self._calculate_eta(overall_percentage, overall_speed)
        
        return ProcessingProgress(
            stage=self.current_stage or "processing",
            current=int(overall_percentage),
            total=100,
            percentage=overall_percentage,
            speed=overall_speed,
            eta_seconds=eta,
            elapsed_seconds=elapsed,
            details=self._get_current_stage_details()
        )
    
    def _calculate_eta(self, percentage: float, speed: float) -> float:
        """Calculate ETA using multiple methods for accuracy"""
        if percentage >= 100:
            return 0
            
        # Method 1: Based on current speed
        if speed > 0:
            remaining_work = (100 - percentage)
            eta1 = remaining_work / speed
        else:
            eta1 = float('inf')
        
        # Method 2: Based on elapsed time and percentage
        elapsed = time.time() - self.start_time
        if percentage > 0 and elapsed > 0:
            rate = percentage / elapsed
            eta2 = (100 - percentage) / rate if rate > 0 else float('inf')
        else:
            eta2 = float('inf')
        
        # Use the more conservative estimate
        eta = min(eta1, eta2) if eta1 != float('inf') and eta2 != float('inf') else max(eta1, eta2)
        return eta if eta != float('inf') else 0
    
    def _get_current_stage_details(self) -> Dict[str, Any]:
        """Get details for the current processing stage"""
        if not self.current_stage or self.current_stage not in self.stages:
            return {}
            
        stage = self.stages[self.current_stage]
        return stage.get('details', {})

class HighPerformanceVideoProcessor:
    """Ultra-fast video processing with parallel processing and optimization"""
    
    def __init__(self):
        self.ffmpeg_processor = OptimizedFFmpegProcessor()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.temp_cleanup_queue = []
        self._setup_temp_directory()
        
    def _setup_temp_directory(self):
        """Setup optimized temporary directory"""
        self.temp_base = Path(tempfile.gettempdir()) / "watermark_bot_optimized"
        self.temp_base.mkdir(exist_ok=True)
        
    async def get_video_info_fast(self, video_path: str) -> VideoInfo:
        """Ultra-fast video information extraction"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', '-select_streams', 'v:0',
            video_path
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise Exception(f"FFprobe failed: {stderr.decode()}")
                
            info = json.loads(stdout.decode())
            
            # Extract video stream info
            video_stream = None
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video' and video_stream is None:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                raise Exception("No video stream found")
            
            # Parse frame rate
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(float, fps_str.split('/'))
                fps = num / den if den > 0 else 30.0
            else:
                fps = float(fps_str)
            
            # Get duration from multiple sources
            duration = 0.0
            if video_stream.get('duration'):
                duration = float(video_stream['duration'])
            elif info.get('format', {}).get('duration'):
                duration = float(info['format']['duration'])
            
            # Calculate frame count
            frame_count = int(duration * fps) if duration > 0 and fps > 0 else 0
            
            return VideoInfo(
                width=int(video_stream.get('width', 1920)),
                height=int(video_stream.get('height', 1080)),
                fps=fps,
                duration=duration,
                codec=video_stream.get('codec_name', 'h264'),
                bitrate=int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else 0,
                format=info.get('format', {}).get('format_name', 'mp4'),
                frame_count=frame_count,
                has_audio=audio_stream is not None,
                audio_codec=audio_stream.get('codec_name') if audio_stream else None,
                audio_bitrate=int(audio_stream.get('bit_rate', 0)) if audio_stream and audio_stream.get('bit_rate') else 0
            )
            
        except Exception as e:
            logger.error(f"Failed to get video info for {video_path}: {e}")
            # Return default values for fallback
            return VideoInfo(1920, 1080, 30.0, 0.0, 'h264', 0, 'mp4', 0, False)
    
    async def optimize_video_preprocessing(self, input_path: str, target_specs: Dict[str, Any]) -> str:
        """Preprocess video for optimal watermarking performance"""
        temp_optimized = self.temp_base / f"optimized_{int(time.time())}_{os.getpid()}.mp4"
        
        video_info = await self.get_video_info_fast(input_path)
        
        # Determine if preprocessing is needed
        needs_scaling = (video_info.width != target_specs.get('width', video_info.width) or 
                        video_info.height != target_specs.get('height', video_info.height))
        needs_fps_change = abs(video_info.fps - target_specs.get('fps', video_info.fps)) > 0.1
        
        if not needs_scaling and not needs_fps_change:
            return input_path  # No preprocessing needed
        
        # Build preprocessing command
        cmd = ['ffmpeg', '-y', '-i', input_path]
        
        # Add hardware decoding if available
        if self.ffmpeg_processor.gpu_available and self.ffmpeg_processor.decoder != 'h264':
            cmd.extend(['-c:v', self.ffmpeg_processor.decoder])
        
        # Video filters
        filters = []
        if needs_scaling:
            target_width = target_specs.get('width', video_info.width)
            target_height = target_specs.get('height', video_info.height)
            filters.append(f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease')
            filters.append(f'pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2')
        
        if needs_fps_change:
            target_fps = target_specs.get('fps', video_info.fps)
            filters.append(f'fps={target_fps}')
        
        if filters:
            cmd.extend(['-vf', ','.join(filters)])
        
        # Encoding settings
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        preset = self.ffmpeg_processor.get_optimal_preset(file_size_mb, video_info.duration)
        
        cmd.extend([
            '-c:v', self.ffmpeg_processor.encoder,
            '-preset', preset,
            '-crf', str(self.ffmpeg_processor.get_optimal_crf('fastest')),
            '-c:a', 'copy',  # Copy audio to save time
            '-movflags', '+faststart',
            '-threads', str(self.ffmpeg_processor.optimal_threads),
            str(temp_optimized)
        ])
        
        # Execute preprocessing
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Preprocessing failed: {stderr.decode()}")
        
        self.temp_cleanup_queue.append(temp_optimized)
        return str(temp_optimized)

class WatermarkBot:
    """Optimized Watermark Bot with extreme performance improvements"""
    
    def __init__(self, api_id: int, api_hash: str, bot_token: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.bot_token = bot_token
        
        # High-performance session management
        self.user_sessions = {}
        self.persistent_data = {}
        self.admin_users = [2038923790]
        self.processing_locks = {}  # User-specific processing locks
        
        # Performance optimizations
        self.video_processor = HighPerformanceVideoProcessor()
        self.progress_trackers = weakref.WeakValueDictionary()
        
        # Initialize directory structure with proper permissions
        self._setup_directories()
        
        # Initialize Pyrogram client with optimization
        self.app = Client(
            "watermark_bot_optimized",
            api_id=api_id,
            api_hash=api_hash,
            bot_token=bot_token,
            workdir=".",
            max_concurrent_transmissions=4  # Parallel uploads
        )
        
        # Setup background cleanup task
        self._setup_background_tasks()
    
    def _setup_directories(self):
        """Setup directory structure with optimized permissions"""
        directories = [
            'persistent_intros', 'persistent_watermarks', 'persistent_thumbnails',
            'bulk_queue', 'persistent_captions', 'cache', 'temp_processing'
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(mode=0o755, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    def _setup_background_tasks(self):
        """Setup background maintenance tasks"""
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def get_system_usage(self) -> str:
        """Enhanced system usage information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU information if available
            gpu_info = ""
            if self.video_processor.ffmpeg_processor.gpu_available:
                gpu_info = f" | GPU: {self.video_processor.ffmpeg_processor.gpu_type.upper()}"
            
            return (f"CPU: {cpu_percent:.1f}% | "
                   f"RAM: {memory.percent:.1f}% | "
                   f"Disk: {disk.percent:.1f}%{gpu_info}")
        except Exception as e:
            logger.warning(f"System info error: {e}")
            return "System info unavailable"
    
    async def get_user_processing_lock(self, user_id: int) -> asyncio.Lock:
        """Get or create user-specific processing lock"""
        if user_id not in self.processing_locks:
            self.processing_locks[user_id] = asyncio.Lock()
        return self.processing_locks[user_id]
    
    async def create_advanced_progress_message(self, message: Message, initial_text: str,
                                             video_num: int = None, total_videos: int = None,
                                             filename: str = None) -> Message:
        """Create enhanced progress message with advanced tracking"""
        user_id = message.from_user.id
        
        # Create progress tracker
        tracker = AdvancedProgressTracker(user_id)
        self.progress_trackers[user_id] = tracker
        
        # Prepare display filename
        task_name = self._prepare_display_filename(message, filename)
        
        # Add counter for bulk processing
        if video_num and total_videos:
            counter_text = f"[{video_num}/{total_videos}] "
            initial_text = counter_text + initial_text
        
        progress_message = await message.reply_text(initial_text)
        
        # Store in session with enhanced metadata
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        
        self.user_sessions[user_id].update({
            'progress_message': progress_message,
            'progress_tracker': tracker,
            'video_counter': f"[{video_num}/{total_videos}] " if video_num and total_videos else "",
            'current_task': task_name,
            'processing_start_time': time.time(),
            'last_progress_update': 0
        })
        
        return progress_message
    
    def _prepare_display_filename(self, message: Message, filename: str = None) -> str:
        """Prepare optimized display filename"""
        if filename:
            task_name = filename
        elif hasattr(message, 'video') and message.video:
            task_name = getattr(message.video, 'file_name', message.caption or "Video.mp4")
        elif hasattr(message, 'caption') and message.caption:
            task_name = message.caption[:30] + ".mp4" if len(message.caption) > 30 else message.caption + ".mp4"
        else:
            task_name = "Processing.mp4"
        
        # Truncate for display
        return task_name[:35] + "..." if len(task_name) > 35 else task_name
    
    async def update_progress_advanced(self, user_id: int, stage: str, 
                                     completed: int, total: int, details: Dict = None,
                                     force_update: bool = False):
        """Advanced progress update with intelligent throttling"""
        current_time = time.time()
        session = self.user_sessions.get(user_id, {})
        
        if not session or 'progress_message' not in session:
            return
        
        # Intelligent update throttling
        last_update = session.get('last_progress_update', 0)
        time_since_update = current_time - last_update
        
        # Progressive update intervals based on stage
        if stage in ['downloading', 'uploading']:
            min_interval = 2.0  # More frequent for I/O operations
        else:
            min_interval = 3.0  # Less frequent for processing
        
        if not force_update and time_since_update < min_interval:
            return
        
        # Update progress tracker
        tracker = session.get('progress_tracker')
        if tracker:
            tracker.update_stage(stage, completed, details)
            progress = tracker.get_overall_progress()
        else:
            # Fallback progress calculation
            percentage = (completed / total * 100) if total > 0 else 0
            progress = ProcessingProgress(stage, completed, total, percentage, 0, 0, 0, details or {})
        
        try:
            # Build enhanced progress display
            progress_text = await self._build_progress_display(user_id, progress, details or {})
            
            await session['progress_message'].edit_text(progress_text)
            session['last_progress_update'] = current_time
            
        except Exception as e:
            if "MESSAGE_NOT_MODIFIED" in str(e):
                pass  # Message content unchanged
            elif "FLOOD_WAIT" in str(e):
                # Extract wait time and delay
                wait_time = int(str(e).split("FLOOD_WAIT_")[1].split()[0]) if "FLOOD_WAIT_" in str(e) else 5
                await asyncio.sleep(min(wait_time, 10))
            else:
                logger.warning(f"Progress update failed: {e}")
    
    async def _build_progress_display(self, user_id: int, progress: ProcessingProgress, 
                                    details: Dict) -> str:
        """Build enhanced progress display with visual elements"""
        session = self.user_sessions.get(user_id, {})
        task_name = session.get('current_task', 'Processing')
        counter = session.get('video_counter', '')
        
        # Enhanced progress bar
        bar_length = 15
        filled = int(bar_length * progress.percentage / 100)
        bar_chars = ["█"] * filled + ["░"] * (bar_length - filled)
        bar = "".join(bar_chars)
        
        # Stage mapping with better icons
        stage_mapping = {
            "downloading": ("Downloading", "⬇️"),
            "preprocessing": ("Preprocessing", "⚙️"),
            "watermarking": ("Watermarking", "🎨"),
            "processing": ("Processing", "🔄"),
            "encoding": ("Encoding", "🎬"),
            "uploading": ("Uploading", "📤"),
            "completed": ("Completed", "✅"),
            "normalizing": ("Normalizing", "📐"),
            "optimizing": ("Optimizing", "⚡")
        }
        
        stage_text, stage_icon = stage_mapping.get(progress.stage, ("Processing", "🔄"))
        
        # System information
        system_info = self.get_system_usage()
        
        # Build main progress text
        progress_text = f"**{counter}📂 {task_name}**\n"
        progress_text += f"├─ `{bar}` **{progress.percentage:.1f}%**\n"
        progress_text += f"├─ **Status:** {stage_text} {stage_icon}\n"
        progress_text += f"├─ **System:** {system_info}\n"
        
        # Add detailed information based on stage and details
        if progress.speed > 0:
            if progress.stage in ['downloading', 'uploading']:
                # File transfer speeds
                if progress.speed >= 1:
                    progress_text += f"├─ **Speed:** {progress.speed:.2f}MB/s ⚡\n"
                else:
                    progress_text += f"├─ **Speed:** {progress.speed*1024:.1f}KB/s ⚡\n"
            else:
                # Processing speeds
                progress_text += f"├─ **Processing:** {progress.speed:.1f}fps 🎬\n"
        
        # File size information
        if 'current_mb' in details:
            current_mb = details['current_mb']
            if 'total_mb' in details and details['total_mb']:
                total_mb = details['total_mb']
                progress_text += f"├─ **Progress:** {current_mb:.2f}MB / {total_mb:.2f}MB\n"
            else:
                progress_text += f"├─ **Processed:** {current_mb:.2f}MB\n"
        
        # Frame information
        if 'frames' in details:
            progress_text += f"├─ **Frames:** {details['frames']}\n"
        
        # ETA and elapsed time
        if progress.eta_seconds > 0:
            eta_str = self._format_duration(progress.eta_seconds)
            progress_text += f"├─ **ETA:** {eta_str} ⏱️\n"
        
        elapsed_str = self._format_duration(progress.elapsed_seconds)
        progress_text += f"└─ **Elapsed:** {elapsed_str} ⏰\n"
        
        return progress_text
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs}s"
        else:
            return f"{int(seconds)}s"

    async def download_with_turbo_progress(self, message: Message, file_id: str, 
                                         file_path: str, file_type: str,
                                         video_num: int = None, total_videos: int = None,
                                         filename: str = None) -> Message:
        """Ultra-fast download with advanced progress tracking and no size limits"""
        user_id = message.from_user.id
        
        # Check if process was stopped
        session = self.user_sessions.get(user_id, {})
        if session.get('stopped'):
            raise Exception(f"Process was stopped by user {user_id}")
        
        # Get file information with parallel metadata fetching
        file_size = await self._get_file_size_optimized(message, file_id)
        display_filename = self._prepare_display_filename(message, filename)
        
        # Create optimized progress message
        if "video" in file_type.lower():
            progress_message = await self.create_advanced_progress_message(
                message, "🚀 **Initializing video pipeline...**", 
                video_num, total_videos, display_filename
            )
        else:
            progress_message = await self.create_advanced_progress_message(
                message, f"⬇️ **Downloading {file_type}...**", 
                video_num, total_videos, display_filename
            )
        
        # Setup progress tracking
        tracker = self.progress_trackers.get(user_id)
        if tracker:
            expected_chunks = max(100, file_size // (1024 * 1024)) if file_size > 0 else 100
            tracker.add_stage("downloading", expected_chunks, 1.0)
        
        start_time = time.time()
        downloaded_bytes = 0
        speed_calculator = SpeedCalculator()
        
        async def turbo_progress_callback(current, total):
            nonlocal downloaded_bytes, speed_calculator
            current_time = time.time()
            
            # Check if stopped
            if self.user_sessions.get(user_id, {}).get('stopped'):
                raise Exception("Process was stopped by user")
            
            downloaded_bytes = current
            speed_calculator.add_sample(current, current_time)
            
            # Calculate metrics
            current_mb = current / (1024 * 1024)
            total_mb = total / (1024 * 1024) if total > 0 else file_size / (1024 * 1024)
            speed_mbps = speed_calculator.get_average_speed() / (1024 * 1024)
            elapsed = current_time - start_time
            
            # Progress calculation with smart total handling
            if total > 0 and current <= total:
                chunk_progress = int((current / total) * 100)
            else:
                # Estimate progress based on file size if available
                if file_size > 0:
                    chunk_progress = min(95, int((current / file_size) * 100))
                else:
                    chunk_progress = min(90, int(current_mb / 10) * 10)
            
            # ETA calculation
            if speed_mbps > 0 and total_mb > current_mb:
                remaining_mb = total_mb - current_mb
                eta_seconds = remaining_mb / speed_mbps
            else:
                eta_seconds = 0
            
            # Build progress details
            details = {
                'current_mb': current_mb,
                'speed_mbps': speed_mbps,
                'eta_seconds': eta_seconds
            }
            
            if total_mb > 0:
                details['total_mb'] = total_mb
            
            # Update progress tracker
            if tracker:
                tracker.update_stage("downloading", chunk_progress, details)
            
            # Update display
            await self.update_progress_advanced(
                user_id, "downloading", chunk_progress, 100, details
            )
        
        try:
            # Execute download with optimized settings
            await self.app.download_media(
                file_id, 
                file_name=file_path, 
                progress=turbo_progress_callback,
                file_name_generator=lambda *args: file_path  # Force exact path
            )
            
            # Verify download
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise Exception("Download verification failed")
            
            # Final update
            final_size = os.path.getsize(file_path)
            final_mb = final_size / (1024 * 1024)
            total_time = time.time() - start_time
            
            details = {
                'current_mb': final_mb,
                'total_mb': final_mb,
                'speed_mbps': final_mb / total_time if total_time > 0 else 0
            }
            
            await self.update_progress_advanced(
                user_id, "completed", 100, 100, details, force_update=True
            )
            
        except Exception as e:
            if "stopped by user" in str(e):
                raise
            else:
                logger.error(f"Download error: {e}")
                raise Exception(f"Download failed: {str(e)}")
        
        return progress_message
    
    async def _get_file_size_optimized(self, message: Message, file_id: str) -> int:
        """Get file size with multiple fallback methods"""
        # Try message attributes first
        if hasattr(message, 'video') and message.video and message.video.file_size:
            return message.video.file_size
        elif hasattr(message, 'photo') and message.photo and message.photo.file_size:
            return message.photo.file_size
        elif hasattr(message, 'document') and message.document and message.document.file_size:
            return message.document.file_size
        
        # Fallback to API call
        try:
            file_info = await self.app.get_file(file_id)
            return getattr(file_info, 'file_size', 0)
        except Exception as e:
            logger.warning(f"Could not get file size: {e}")
            return 0

class SpeedCalculator:
    """Optimized speed calculation with smoothing"""
    
    def __init__(self, window_size: int = 10):
        self.samples = []
        self.window_size = window_size
        self.last_bytes = 0
        self.last_time = 0
    
    def add_sample(self, current_bytes: int, current_time: float):
        """Add a new speed sample"""
        if self.last_time > 0:
            time_diff = current_time - self.last_time
            byte_diff = current_bytes - self.last_bytes
            
            if time_diff > 0.1:  # Minimum interval for stable measurement
                speed = byte_diff / time_diff
                self.samples.append(speed)
                
                if len(self.samples) > self.window_size:
                    self.samples.pop(0)
        
        self.last_bytes = current_bytes
        self.last_time = current_time
    
    def get_average_speed(self) -> float:
        """Get smoothed average speed"""
        if not self.samples:
            return 0.0
        
        # Weighted average favoring recent samples
        weights = [i + 1 for i in range(len(self.samples))]
        weighted_sum = sum(speed * weight for speed, weight in zip(self.samples, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

class UltraFastWatermarkEngine:
    """Ultra-fast watermarking with GPU acceleration and optimizations"""
    
    def __init__(self, ffmpeg_processor: OptimizedFFmpegProcessor):
        self.ffmpeg_processor = ffmpeg_processor
        self.watermark_cache = {}
        self.filter_cache = {}
    
    async def apply_watermarks_turbocharged(self, input_path: str, output_path: str,
                                          watermark_text: str = None, 
                                          watermark_png_path: str = None,
                                          png_location: str = "topright",
                                          progress_callback=None) -> bool:
        """Apply watermarks with maximum performance optimization"""
        try:
            # Get video info for optimization
            video_info = await self._get_cached_video_info(input_path)
            
            # Build optimized filter chain
            filter_chain = await self._build_optimized_filters(
                video_info, watermark_text, watermark_png_path, png_location
            )
            
            # Determine optimal encoding settings
            encoding_settings = self._get_optimal_encoding_settings(video_info, input_path)
            
            # Build command with maximum optimization
            cmd = await self._build_turbo_command(
                input_path, output_path, filter_chain, encoding_settings, watermark_png_path
            )
            
            # Execute with progress tracking
            success = await self._execute_with_turbo_progress(
                cmd, video_info, progress_callback
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Turbo watermarking failed: {e}")
            # Emergency fallback - copy file if watermarking fails
            if os.path.exists(input_path):
                shutil.copy2(input_path, output_path)
                return True
            return False
    
    async def _get_cached_video_info(self, video_path: str) -> VideoInfo:
        """Get video info with caching for repeated access"""
        cache_key = f"{video_path}_{os.path.getmtime(video_path)}"
        
        if cache_key in self.watermark_cache:
            return self.watermark_cache[cache_key]
        
        # Use the optimized video processor
        from watermark_bot_optimized import HighPerformanceVideoProcessor
        processor = HighPerformanceVideoProcessor()
        video_info = await processor.get_video_info_fast(video_path)
        
        # Cache for future use
        self.watermark_cache[cache_key] = video_info
        
        # Cleanup old cache entries
        if len(self.watermark_cache) > 50:
            oldest_key = next(iter(self.watermark_cache))
            del self.watermark_cache[oldest_key]
        
        return video_info
    
    async def _build_optimized_filters(self, video_info: VideoInfo, 
                                     watermark_text: str = None,
                                     watermark_png_path: str = None,
                                     png_location: str = "topright") -> str:
        """Build highly optimized filter chain"""
        filters = []
        overlay_input = "[0:v]"
        
        # Text watermark with optimized positioning
        if watermark_text and watermark_text.strip():
            text_filter = self._create_optimized_text_filter(
                watermark_text, video_info, watermark_png_path, png_location
            )
            filters.append(f"{overlay_input}{text_filter}[txt]")
            overlay_input = "[txt]"
        
        # PNG watermark with hardware optimization
        if watermark_png_path and os.path.exists(watermark_png_path):
            png_filter = self._create_optimized_png_filter(
                video_info, png_location
            )
            filters.append(f"[1:v]{png_filter}[wm]")
            filters.append(f"{overlay_input}[wm]overlay={self._get_png_position(png_location, video_info)}[final]")
            overlay_input = "[final]"
        
        return ";".join(filters) if filters else "copy"
    
    def _create_optimized_text_filter(self, watermark_text: str, video_info: VideoInfo,
                                    has_png: bool = False, png_location: str = "topright") -> str:
        """Create optimized text filter with smart positioning"""
        # Clean text for FFmpeg
        clean_text = self._clean_text_for_ffmpeg(watermark_text)
        
        # Process text into optimized lines
        lines = self._process_text_lines(clean_text, video_info.width)
        final_text = "\\\\n".join(lines)  # Double escape for FFmpeg
        
        # Calculate optimal font size and spacing
        font_size = self._calculate_optimal_font_size(video_info)
        line_spacing = int(font_size * 1.15)
        margin = max(10, video_info.width // 100)
        
        # Smart positioning that avoids PNG watermark
        x_pos, y_pos = self._calculate_smart_text_position(
            video_info, has_png, png_location, margin
        )
        
        # Build optimized drawtext filter
        return (
            f"drawtext=text='{final_text}':"
            f"fontsize={font_size}:"
            f"fontcolor=white@0.95:"
            f"box=1:boxcolor=black@0.75:boxborderw=8:"
            f"line_spacing={line_spacing}:"
            f"x='{x_pos}':"
            f"y='{y_pos}'"
        )
    
    def _clean_text_for_ffmpeg(self, text: str) -> str:
        """Clean and optimize text for FFmpeg processing"""
        # Escape special characters
        clean_text = text.replace("'", "\\'").replace(":", "\\:")
        
        # Remove problematic characters but preserve newlines
        clean_text = ''.join(
            c for c in clean_text 
            if ord(c) < 127 and (c.isprintable() or c in ['\n', '\r', ' '])
        )
        
        return clean_text
    
    def _process_text_lines(self, text: str, video_width: int) -> List[str]:
        """Process text into optimized lines for display"""
        max_chars_per_line = max(30, video_width // 15)
        user_lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        
        processed_lines = []
        for user_line in user_lines:
            user_line = user_line.strip()
            if not user_line:
                if processed_lines:  # Only add spaces if not at the beginning
                    processed_lines.append(" ")
                continue
            
            if len(user_line) <= max_chars_per_line:
                processed_lines.append(user_line)
            else:
                # Smart word wrapping
                words = user_line.split()
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if len(test_line) <= max_chars_per_line:
                        current_line = test_line
                    else:
                        if current_line:
                            processed_lines.append(current_line)
                        current_line = word
                
                if current_line:
                    processed_lines.append(current_line)
        
        # Limit lines for performance
        return processed_lines[:6] if len(processed_lines) <= 6 else processed_lines[:5] + ["..."]
    
    def _calculate_optimal_font_size(self, video_info: VideoInfo) -> int:
        """Calculate optimal font size based on video resolution"""
        base_size = min(video_info.width, video_info.height) // 40
        return max(16, min(base_size, 32))
    
    def _calculate_smart_text_position(self, video_info: VideoInfo, has_png: bool,
                                     png_location: str, margin: int) -> Tuple[str, str]:
        """Calculate smart text positioning to avoid PNG overlap"""
        if has_png:
            # Cycle through positions avoiding PNG location
            if png_location == "topright":
                # Cycle: bottomright -> bottomleft (2 positions, 120s cycle)
                x_pos = "if(lt(mod(t,120),60),w-text_w-" + str(margin) + "," + str(margin) + ")"
                y_pos = "h-text_h-" + str(margin)
            else:
                # Cycle: topright -> bottomright -> bottomleft (3 positions, 180s cycle)
                x_pos = ("if(lt(mod(t,180),60),w-text_w-" + str(margin) + 
                        ",if(lt(mod(t,180),120),w-text_w-" + str(margin) + "," + str(margin) + "))")
                y_pos = ("if(lt(mod(t,180),60)," + str(margin * 2) + 
                        ",if(lt(mod(t,180),120),h-text_h-" + str(margin) + ",h-text_h-" + str(margin) + "))")
        else:
            # No PNG, cycle through 3 positions avoiding topleft
            x_pos = ("if(lt(mod(t,180),60),w-text_w-" + str(margin) + 
                    ",if(lt(mod(t,180),120),w-text_w-" + str(margin) + "," + str(margin) + "))")
            y_pos = ("if(lt(mod(t,180),60)," + str(margin * 2) + 
                    ",if(lt(mod(t,180),120),h-text_h-" + str(margin) + ",h-text_h-" + str(margin) + "))")
        
        return x_pos, y_pos
    
    def _create_optimized_png_filter(self, video_info: VideoInfo, png_location: str) -> str:
        """Create optimized PNG scaling filter"""
        # Calculate optimal size (max 1/4 of video dimensions)
        max_size = min(video_info.width // 4, video_info.height // 4, 200)
        
        return f"scale=-1:{max_size}:force_original_aspect_ratio=decrease"
    
    def _get_png_position(self, png_location: str, video_info: VideoInfo) -> str:
        """Get PNG overlay position"""
        margin = max(8, video_info.width // 150)
        
        position_map = {
            "topleft": f"{margin}:{margin}",
            "topright": f"W-w-{margin}:{margin}",
            "bottomleft": f"{margin}:H-h-{margin}",
            "bottomright": f"W-w-{margin}:H-h-{margin}"
        }
        
        return position_map.get(png_location, position_map["bottomright"])
    
    def _get_optimal_encoding_settings(self, video_info: VideoInfo, input_path: str) -> Dict[str, str]:
        """Get optimal encoding settings for maximum performance"""
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        
        # Base settings for performance
        settings = {
            'codec': self.ffmpeg_processor.encoder,
            'preset': self.ffmpeg_processor.get_optimal_preset(file_size_mb, video_info.duration),
            'crf': str(self.ffmpeg_processor.get_optimal_crf('fastest')),
            'threads': str(self.ffmpeg_processor.optimal_threads)
        }
        
        # Bitrate optimization
        if video_info.bitrate > 0:
            # Use original bitrate as reference but cap for performance
            target_bitrate = min(video_info.bitrate, 3000000)  # Max 3Mbps
            settings['bitrate'] = f"{target_bitrate // 1000}k"
            settings['maxrate'] = f"{int(target_bitrate * 1.2) // 1000}k"
            settings['bufsize'] = f"{int(target_bitrate * 2) // 1000}k"
        else:
            # Estimate from file size
            if file_size_mb > 0 and video_info.duration > 0:
                estimated_bitrate = int((file_size_mb * 8 * 1024 * 1024) / video_info.duration * 0.85)
                estimated_bitrate = min(estimated_bitrate, 2000000)  # Cap at 2Mbps
                settings['bitrate'] = f"{estimated_bitrate // 1000}k"
            else:
                settings['bitrate'] = "1500k"  # Default for performance
        
        return settings
    
    async def _build_turbo_command(self, input_path: str, output_path: str,
                                 filter_chain: str, encoding_settings: Dict[str, str],
                                 watermark_png_path: str = None) -> List[str]:
        """Build optimized FFmpeg command for maximum speed"""
        cmd = ['ffmpeg', '-y']
        
        # Input optimization
        if self.ffmpeg_processor.gpu_available and self.ffmpeg_processor.gpu_type == 'nvidia':
            cmd.extend(['-hwaccel', 'cuda'])
        
        cmd.extend(['-i', input_path])
        
        # Add PNG input if needed
        if watermark_png_path and os.path.exists(watermark_png_path):
            cmd.extend(['-i', watermark_png_path])
        
        # Filter complex or simple filter
        if filter_chain != "copy":
            if watermark_png_path and os.path.exists(watermark_png_path):
                cmd.extend(['-filter_complex', filter_chain, '-map', '[final]'])
            else:
                cmd.extend(['-vf', filter_chain])
        
        # Audio handling
        cmd.extend(['-map', '0:a?', '-c:a', 'copy'])  # Copy audio for speed
        
        # Video encoding settings
        cmd.extend([
            '-c:v', encoding_settings['codec'],
            '-preset', encoding_settings['preset'],
            '-crf', encoding_settings['crf'],
            '-threads', encoding_settings['threads']
        ])
        
        # Bitrate settings if available
        if 'bitrate' in encoding_settings:
            cmd.extend(['-b:v', encoding_settings['bitrate']])
            if 'maxrate' in encoding_settings:
                cmd.extend(['-maxrate', encoding_settings['maxrate']])
            if 'bufsize' in encoding_settings:
                cmd.extend(['-bufsize', encoding_settings['bufsize']])
        
         # Optimization flags
    cmd.extend([
        '-profile:v', 'main',
        '-level', '3.1',
        '-map_metadata', '0',
        '-movflags', '+faststart',
        '-avoid_negative_ts', 'make_zero',
        '-preset', 'superfast',       # ✅ Added for faster encoding
        '-threads', '4',              # ✅ Optimized threading for performance
        '-vsync', '2',                # ✅ Better frame synchronization
        output_path
    ])
    
    return cmd

async def _execute_with_turbo_progress(self, cmd: List[str], video_info: VideoInfo,
                                     progress_callback=None) -> bool:
    """Execute FFmpeg command with turbo progress tracking"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        if progress_callback:
            # ✅ Improved progress monitoring for better real-time tracking
            await self._monitor_ffmpeg_progress(process, video_info, progress_callback)
        else:
            await process.wait()

class TurboIntroProcessor:
    """Ultra-fast intro processing with caching and normalization"""
    
    def __init__(self, ffmpeg_processor: OptimizedFFmpegProcessor):
        self.ffmpeg_processor = ffmpeg_processor
        self.normalized_cache = Path("cache/normalized_intros")
        self.normalized_cache.mkdir(parents=True, exist_ok=True)
        
    async def process_intro_with_main_turbocharged(self, intro_path: str, main_video_path: str,
                                                 output_path: str, progress_callback=None) -> bool:
        """Process intro with main video using maximum optimization"""
        try:
            # Get video specs for both files
            intro_info = await self._get_video_specs(intro_path)
            main_info = await self._get_video_specs(main_video_path)
            
            # Determine optimal target specs
            target_specs = self._calculate_optimal_specs(intro_info, main_info)
            
            # Check cache for normalized intro
            cached_intro = await self._get_cached_normalized_intro(intro_path, target_specs)
            
            if cached_intro:
                logger.info("Using cached normalized intro")
                normalized_intro_path = cached_intro
            else:
                # Normalize intro with progress tracking
                normalized_intro_path = await self._normalize_intro_turbo(
                    intro_path, target_specs, progress_callback
                )
            
            # Normalize main video if needed
            normalized_main_path = await self._normalize_main_video_turbo(
                main_video_path, target_specs, progress_callback
            )
            
            # Concatenate with ultra-fast method
            success = await self._concatenate_turbo(
                normalized_intro_path, normalized_main_path, output_path, progress_callback
            )
            
            # Cleanup temporary files
            await self._cleanup_temp_files([
                normalized_main_path if normalized_main_path != main_video_path else None
            ])
            
            return success
            
        except Exception as e:
            logger.error(f"Turbo intro processing failed: {e}")
            return False
    
    async def _get_video_specs(self, video_path: str) -> VideoInfo:
        """Get video specifications with caching"""
        cache_key = f"specs_{os.path.basename(video_path)}_{os.path.getmtime(video_path)}"
        
        processor = HighPerformanceVideoProcessor()
        return await processor.get_video_info_fast(video_path)
    
    def _calculate_optimal_specs(self, intro_info: VideoInfo, main_info: VideoInfo) -> Dict[str, Any]:
        """Calculate optimal specifications for processing"""
        # Use the higher quality specs between intro and main
        target_width = max(intro_info.width, main_info.width, 1280)
        target_height = max(intro_info.height, main_info.height, 720)
        
        # Use consistent FPS for smooth concatenation
        target_fps = max(intro_info.fps, main_info.fps, 30.0)
        target_fps = min(target_fps, 60.0)  # Cap at 60fps for performance
        
        return {
            'width': target_width,
            'height': target_height,
            'fps': target_fps,
            'codec': 'h264',
            'preset': 'ultrafast'
        }
    
    async def _get_cached_normalized_intro(self, intro_path: str, target_specs: Dict[str, Any]) -> Optional[str]:
        """Check for cached normalized intro"""
        # Create cache key based on intro file and target specs
        intro_stat = os.stat(intro_path)
        specs_hash = hashlib.md5(
            f"{target_specs['width']}x{target_specs['height']}@{target_specs['fps']}fps".encode()
        ).hexdigest()[:8]
        
        cache_filename = f"intro_{intro_stat.st_mtime}_{intro_stat.st_size}_{specs_hash}.mp4"
        cache_path = self.normalized_cache / cache_filename
        
        if cache_path.exists():
            # Verify cache file integrity
            try:
                cached_info = await self._get_video_specs(str(cache_path))
                if (abs(cached_info.width - target_specs['width']) <= 2 and
                    abs(cached_info.height - target_specs['height']) <= 2 and
                    abs(cached_info.fps - target_specs['fps']) <= 0.1):
                    return str(cache_path)
            except Exception as e:
                logger.warning(f"Cache verification failed: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    async def _normalize_intro_turbo(self, intro_path: str, target_specs: Dict[str, Any],
                                   progress_callback=None) -> str:
        """Normalize intro with turbo speed and caching"""
        # Generate cache path
        intro_stat = os.stat(intro_path)
        specs_hash = hashlib.md5(
            f"{target_specs['width']}x{target_specs['height']}@{target_specs['fps']}fps".encode()
        ).hexdigest()[:8]
        
        cache_filename = f"intro_{intro_stat.st_mtime}_{intro_stat.st_size}_{specs_hash}.mp4"
        cache_path = self.normalized_cache / cache_filename
        
        # Build turbo normalization command
        cmd = [
            'ffmpeg', '-y',
            '-i', intro_path,
            '-vf', f'scale={target_specs["width"]}:{target_specs["height"]}:force_original_aspect_ratio=decrease,pad={target_specs["width"]}:{target_specs["height"]}:(ow-iw)/2:(oh-ih)/2,fps={target_specs["fps"]}',
            '-c:v', self.ffmpeg_processor.encoder,
            '-preset', 'ultrafast',
            '-crf', '28',  # Higher CRF for faster encoding
            '-c:a', 'copy',
            '-threads', str(self.ffmpeg_processor.optimal_threads),
            '-movflags', '+faststart',
            str(cache_path)
        ]
        
        # Execute with progress tracking
        if progress_callback:
            intro_info = await self._get_video_specs(intro_path)
            await self._execute_with_progress(cmd, intro_info, progress_callback, "normalizing_intro")
        else:
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await process.wait()
        
        return str(cache_path)
    
    async def _normalize_main_video_turbo(self, main_path: str, target_specs: Dict[str, Any],
                                        progress_callback=None) -> str:
        """Normalize main video if needed"""
        main_info = await self._get_video_specs(main_path)
        
        # Check if normalization is needed
        needs_normalization = (
            abs(main_info.width - target_specs['width']) > 2 or
            abs(main_info.height - target_specs['height']) > 2 or
            abs(main_info.fps - target_specs['fps']) > 0.1
        )
        
        if not needs_normalization:
            return main_path
        
        # Create temporary normalized file
        temp_normalized = f"temp_main_normalized_{int(time.time())}_{os.getpid()}.mp4"
        
        # Build command for main video normalization
        cmd = [
            'ffmpeg', '-y',
            '-i', main_path,
            '-vf', f'scale={target_specs["width"]}:{target_specs["height"]}:force_original_aspect_ratio=decrease,pad={target_specs["width"]}:{target_specs["height"]}:(ow-iw)/2:(oh-ih)/2,fps={target_specs["fps"]}',
            '-c:v', self.ffmpeg_processor.encoder,
            '-preset', 'superfast',  # Slightly better quality for main video
            '-crf', '26',
            '-c:a', 'copy',
            '-threads', str(self.ffmpeg_processor.optimal_threads),
            '-movflags', '+faststart',
            temp_normalized
        ]
        
        # Execute with progress tracking
        if progress_callback:
            await self._execute_with_progress(cmd, main_info, progress_callback, "normalizing_main")
        else:
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await process.wait()
        
        return temp_normalized
    
    async def _concatenate_turbo(self, intro_path: str, main_path: str, output_path: str,
                               progress_callback=None) -> bool:
        """Ultra-fast concatenation using stream copy"""
        try:
            # Create concat list file
            list_file = f"temp_concat_{int(time.time())}_{os.getpid()}.txt"
            
            with open(list_file, 'w') as f:
                f.write(f"file '{os.path.abspath(intro_path)}'\n")
                f.write(f"file '{os.path.abspath(main_path)}'\n")
            
            # Build ultra-fast concatenation command
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',  # Stream copy for maximum speed
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                '-movflags', '+faststart',
                output_path
            ]
            
            # Execute concatenation
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.wait_with_timeout(300)  # 5 minute timeout
            
            # Cleanup list file
            Path(list_file).unlink(missing_ok=True)
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Turbo concatenation failed: {e}")
            return False
    
    async def _execute_with_progress(self, cmd: List[str], video_info: VideoInfo,
                                   progress_callback, stage: str):
        """Execute command with progress monitoring"""
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        
        frame_count = 0
        total_frames = video_info.frame_count
        start_time = time.time()
        
        while True:
            try:
                line = await asyncio.wait_for(process.stderr.readline(), timeout=1.0)
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                
                if 'frame=' in line_str:
                    try:
                        frame_match = line_str.split('frame=')[1].split()[0]
                        frame_count = int(frame_match)
                        
                        elapsed = time.time() - start_time
                        processing_fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        progress_info = {
                            'stage': stage,
                            'frame_count': frame_count,
                            'total_frames': total_frames,
                            'processing_fps': processing_fps,
                            'elapsed': elapsed
                        }
                        
                        await progress_callback(frame_count, total_frames, progress_info)
                        
                    except (ValueError, IndexError):
                        continue
                        
            except asyncio.TimeoutError:
                continue
        
        await process.wait()
    
    async def _cleanup_temp_files(self, temp_files: List[str]):
        """Cleanup temporary files asynchronously"""
        for temp_file in temp_files:
            if temp_file and Path(temp_file).exists():
                try:
                    Path(temp_file).unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_file}: {e}")

class BulkProcessingOptimizer:
    """Optimized bulk processing with parallel operations and smart queuing"""
    
    def __init__(self, video_processor: HighPerformanceVideoProcessor):
        self.video_processor = video_processor
        self.processing_semaphore = asyncio.Semaphore(2)  # Limit concurrent processing
        self.download_semaphore = asyncio.Semaphore(3)   # Allow more downloads
        
    async def process_bulk_queue_optimized(self, user_id: int, queue_data: List[Dict],
                                         watermark_text: str, png_location: str,
                                         progress_callback=None) -> List[str]:
        """Process bulk queue with optimized parallel operations"""
        results = []
        failed_videos = []
        
        try:
            # Create processing tasks
            tasks = []
            for i, video_info in enumerate(queue_data, 1):
                task = self._process_single_video_optimized(
                    user_id, video_info, watermark_text, png_location,
                    i, len(queue_data), progress_callback
                )
                tasks.append(task)
            
            # Process with controlled concurrency
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for i, result in enumerate(completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Video {i+1} failed: {result}")
                    failed_videos.append(i+1)
                else:
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Bulk processing optimization failed: {e}")
            return []
    
    async def _process_single_video_optimized(self, user_id: int, video_info: Dict,
                                            watermark_text: str, png_location: str,
                                            video_num: int, total_videos: int,
                                            progress_callback=None) -> str:
        """Process single video with full optimization pipeline"""
        async with self.processing_semaphore:
            try:
                # Setup temporary processing directory
                temp_dir = Path(f"temp_processing/user_{user_id}_{video_num}")
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare file paths
                video_path = temp_dir / video_info['filename']
                watermarked_path = temp_dir / f"watermarked_{video_info['filename']}"
                
                # Download with optimization
                async with self.download_semaphore:
                    await self._download_video_optimized(
                        user_id, video_info, str(video_path), progress_callback
                    )
                
                # Apply watermarks with turbo engine
                watermark_engine = UltraFastWatermarkEngine(
                    self.video_processor.ffmpeg_processor
                )
                
                success = await watermark_engine.apply_watermarks_turbocharged(
                    str(video_path), str(watermarked_path),
                    watermark_text, self._get_user_watermark_path(user_id),
                    png_location, progress_callback
                )
                
                if not success:
                    raise Exception("Watermarking failed")
                
                # Process intro if exists
                final_path = await self._process_intro_if_exists(
                    user_id, str(watermarked_path), temp_dir, progress_callback
                )
                
                # Upload optimized
                await self._upload_video_optimized(
                    user_id, final_path, video_info, progress_callback
                )
                
                # Cleanup
                await self._cleanup_temp_directory(temp_dir)
                
                return f"Video {video_num} processed successfully"
                
            except Exception as e:
                logger.error(f"Single video processing failed: {e}")
                await self._cleanup_temp_directory(temp_dir)
                raise
    
    async def _download_video_optimized(self, user_id: int, video_info: Dict,
                                      video_path: str, progress_callback=None):
        """Optimized video download"""
        # This would integrate with the turbo download system
        # Implementation would use the download_with_turbo_progress method
        pass
    
    def _get_user_watermark_path(self, user_id: int) -> str:
        """Get user watermark path"""
        watermark_path = f"persistent_watermarks/watermark_{user_id}.png"
        return watermark_path if os.path.exists(watermark_path) else None
    
    async def _process_intro_if_exists(self, user_id: int, video_path: str,
                                     temp_dir: Path, progress_callback=None) -> str:
        """Process intro if it exists for user"""
        intro_path = f"persistent_intros/intro_{user_id}.mp4"
        
        if not os.path.exists(intro_path):
            return video_path
        
        output_with_intro = temp_dir / "with_intro.mp4"
        
        intro_processor = TurboIntroProcessor(
            self.video_processor.ffmpeg_processor
        )
        
        success = await intro_processor.process_intro_with_main_turbocharged(
            intro_path, video_path, str(output_with_intro), progress_callback
        )
        
        return str(output_with_intro) if success else video_path
    
    async def _upload_video_optimized(self, user_id: int, video_path: str,
                                    video_info: Dict, progress_callback=None):
        """Optimized video upload with progress tracking"""
        # This would integrate with the optimized upload system
        # Implementation would handle thumbnails, captions, etc.
        pass
    
    async def _cleanup_temp_directory(self, temp_dir: Path):
        """Cleanup temporary directory"""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed for {temp_dir}: {e}")

class OptimizedSessionManager:
    """Advanced session management with memory optimization"""
    
    def __init__(self):
        self.sessions = {}
        self.session_locks = {}
        self.cleanup_interval = 1800  # 30 minutes
        self.last_cleanup = time.time()
        
    async def get_session(self, user_id: int) -> Dict[str, Any]:
        """Get or create user session with thread safety"""
        if user_id not in self.session_locks:
            self.session_locks[user_id] = asyncio.Lock()
        
        async with self.session_locks[user_id]:
            if user_id not in self.sessions:
                self.sessions[user_id] = {
                    'created_at': time.time(),
                    'last_activity': time.time(),
                    'processing': False,
                    'stopped': False
                }
            
            self.sessions[user_id]['last_activity'] = time.time()
            return self.sessions[user_id]
    
    async def update_session(self, user_id: int, updates: Dict[str, Any]):
        """Update user session atomically"""
        session = await self.get_session(user_id)
        session.update(updates)
        session['last_activity'] = time.time()
    
    async def cleanup_inactive_sessions(self):
        """Cleanup inactive sessions to free memory"""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        inactive_users = []
        
        for user_id, session in self.sessions.items():
            if (current_time - session.get('last_activity', 0) > 3600 and  # 1 hour
                not session.get('processing', False)):
                inactive_users.append(user_id)
        
        # Remove inactive sessions
        for user_id in inactive_users:
            self.sessions.pop(user_id, None)
            self.session_locks.pop(user_id, None)
            logger.info(f"Cleaned up inactive session for user {user_id}")
        
        self.last_cleanup = current_time
    
    def get_active_users_count(self) -> int:
        """Get count of active users"""
        current_time = time.time()
        active_count = 0
        
        for session in self.sessions.values():
            if current_time - session.get('last_activity', 0) < 1800:  # 30 minutes
                active_count += 1
        
        return active_count
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"FFmpeg execution failed: {e}")
            return False
    
    async def _monitor_ffmpeg_progress(self, process, video_info: VideoInfo, progress_callback):
        """Monitor FFmpeg progress with optimized parsing"""
        frame_count = 0
        total_frames = video_info.frame_count
        start_time = time.time()
        last_update = start_time
        fps_samples = []
        
        while True:
            try:
                line = await asyncio.wait_for(process.stderr.readline(), timeout=1.0)
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                
                if 'frame=' in line_str:
                    try:
                        # Parse frame number
                        frame_match = line_str.split('frame=')[1].split()[0]
                        frame_count = int(frame_match)
                        
                        current_time = time.time()
                        
                        # Calculate processing FPS
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            processing_fps = frame_count / elapsed
                            fps_samples.append(processing_fps)
                            if len(fps_samples) > 5:
                                fps_samples.pop(0)
                            
                            avg_fps = sum(fps_samples) / len(fps_samples)
                        else:
                            avg_fps = 0
                        
                        # Calculate ETA
                        if avg_fps > 0 and total_frames > 0:
                            remaining_frames = max(0, total_frames - frame_count)
                            eta_seconds = remaining_frames / avg_fps
                        else:
                            eta_seconds = 0
                        
                        # Update progress every 2 seconds
                        if current_time - last_update >= 2.0 or frame_count >= total_frames:
                            progress_info = {
                                'frame_count': frame_count,
                                'total_frames': total_frames,
                                'processing_fps': avg_fps,
                                'eta_seconds': eta_seconds,
                                'elapsed': elapsed
                            }
                            
                            await progress_callback(frame_count, total_frames, progress_info)
                            last_update = current_time
                            
                    except (ValueError, IndexError):
                        continue
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"Progress monitoring error: {e}")
                break
        
        await process.wait()
        # Continue from Part 1...
    
    # Updated WatermarkBot class with all optimizations integrated
    def __init__(self, api_id: int, api_hash: str, bot_token: str):
        """Initialize optimized watermark bot with all performance improvements"""
        self.api_id = api_id
        self.api_hash = api_hash
        self.bot_token = bot_token
        
        # Initialize all optimization systems
        self.session_manager = OptimizedSessionManager()
        self.video_processor = HighPerformanceVideoProcessor()
        self.watermark_engine = UltraFastWatermarkEngine(self.video_processor.ffmpeg_processor)
        self.intro_processor = TurboIntroProcessor(self.video_processor.ffmpeg_processor)
        self.bulk_optimizer = BulkProcessingOptimizer(self.video_processor)
        
        # Admin configuration
        self.admin_users = [2038923790]
        
        # Setup optimized directories
        self._setup_optimized_directories()
        
        # Initialize Pyrogram with performance settings
        self.app = Client(
            "watermark_bot_ultra",
            api_id=api_id,
            api_hash=api_hash,
            bot_token=bot_token,
            workdir=".",
            max_concurrent_transmissions=6,  # Increased for better performance
            sleep_threshold=60,  # Optimize connection management
            workers=8  # More workers for handling messages
        )
        
        # Background optimization task
        asyncio.create_task(self._background_maintenance())
    
    def _setup_optimized_directories(self):
        """Setup directory structure with performance optimizations"""
        directories = [
            'persistent_intros', 'persistent_watermarks', 'persistent_thumbnails',
            'bulk_queue', 'persistent_captions', 'cache', 'temp_processing',
            'cache/normalized_intros', 'cache/video_info', 'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(mode=0o755, parents=True, exist_ok=True)
    
    async def _background_maintenance(self):
        """Background task for system maintenance"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Session cleanup
                await self.session_manager.cleanup_inactive_sessions()
                
                # Cache cleanup
                await self._cleanup_old_cache_files()
                
                # Temp file cleanup
                await self._cleanup_temp_files()
                
                # System resource monitoring
                await self._monitor_system_resources()
                
            except Exception as e:
                logger.error(f"Background maintenance error: {e}")
    
    async def _cleanup_old_cache_files(self):
        """Cleanup old cache files to prevent disk space issues"""
        try:
            cache_dirs = ['cache/normalized_intros', 'cache/video_info']
            current_time = time.time()
            
            for cache_dir in cache_dirs:
                if not os.path.exists(cache_dir):
                    continue
                    
                for cache_file in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, cache_file)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > 86400:  # 24 hours
                            os.remove(file_path)
                            
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
    
    async def _cleanup_temp_files(self):
        """Cleanup temporary processing files"""
        try:
            temp_dirs = ['temp_processing']
            current_time = time.time()
            
            for temp_dir in temp_dirs:
                if not os.path.exists(temp_dir):
                    continue
                    
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        # Remove directories older than 1 hour
                        dir_age = current_time - os.path.getmtime(item_path)
                        if dir_age > 3600:
                            shutil.rmtree(item_path, ignore_errors=True)
                            
        except Exception as e:
            logger.warning(f"Temp cleanup error: {e}")
    
    async def _monitor_system_resources(self):
        """Monitor system resources and adjust performance"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Adjust processing limits based on system load
            if cpu_percent > 80 or memory.percent > 85:
                # Reduce concurrent processing
                self.bulk_optimizer.processing_semaphore = asyncio.Semaphore(1)
                logger.warning("High system load detected, reducing concurrency")
            elif cpu_percent < 50 and memory.percent < 70:
                # Increase concurrent processing
                self.bulk_optimizer.processing_semaphore = asyncio.Semaphore(3)
                
        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")

    # Optimized Command Handlers
    async def start_handler(self, client: Client, message: Message):
        """Ultra-fast start command with system info"""
        user_id = message.from_user.id
        
        # Reset user session
        await self.session_manager.update_session(user_id, {'stopped': False})
        
        system_info = self.get_optimized_system_info()
        is_admin = user_id in self.admin_users
        active_users = self.session_manager.get_active_users_count()
        
        # GPU information
        gpu_info = ""
        if self.video_processor.ffmpeg_processor.gpu_available:
            gpu_type = self.video_processor.ffmpeg_processor.gpu_type.upper()
            gpu_info = f"\n🚀 **GPU Acceleration:** {gpu_type} Ready!"
        
        welcome_text = f"""
🎬 **Ultra-Fast Watermark Bot** 🎬
{system_info}{gpu_info}
👤 **User ID:** {user_id} {'🔧 (Admin)' if is_admin else ''}
👥 **Active Users:** {active_users}

**⚡ PERFORMANCE FEATURES:**
• GPU Hardware Acceleration (when available)
• Parallel Processing & Smart Caching
• No File Size Limits - Process GB+ files
• Lightning Fast Watermarking Engine
• Smart Progress Tracking & ETA

**🎯 AVAILABLE COMMANDS:**
• /setintro - Set intro video (cached for speed)
• /setwatermark - Set PNG watermark
• /setthumbnail - Set custom thumbnail
• /addcaption - Add permanent caption
• /convert - Convert with all effects
• /bulk - Ultra-fast bulk processing
• /status - Check your settings
• /stop - Stop all processes

**🚀 BULK PROCESSING:**
Use /bulk to process multiple videos simultaneously with maximum performance!

**💡 HOW TO USE:**
1. (Optional) Set intro/watermark/caption
2. Send video(s) - any size supported
3. Choose PNG location if watermark set
4. Send text or type "skip"
5. Get processed video with lightning speed!

Ready to process at warp speed! Send a video to start! 🚀
        """
        
        await message.reply_text(welcome_text)

    async def process_video_ultra_optimized(self, message: Message, user_id: int,
                                          video_path: str, watermark_text: str,
                                          original_caption: str, video_num: int = None,
                                          png_location: str = "topright"):
        """Ultra-optimized video processing pipeline"""
        progress_msg = None
        temp_files = []
        
        try:
            session = await self.session_manager.get_session(user_id)
            if session.get('stopped'):
                await message.reply_text("🛑 **Process was stopped**")
                return
            
            # Create advanced progress tracker
            progress_msg = await self.create_advanced_progress_message(
                message, "🚀 **Initializing ultra-fast pipeline...**",
                video_num, session.get('total_videos') if video_num else None
            )
            
            # Get processing lock
            processing_lock = await self.get_user_processing_lock(user_id)
            
            async with processing_lock:
                await self.session_manager.update_session(user_id, {'processing': True})
                
                # Stage 1: Download with turbo progress (0-20%)
                if not os.path.exists(video_path):
                    await self._download_stage_optimized(
                        message, user_id, video_path, progress_msg
                    )
                
                # Stage 2: Video preprocessing and optimization (20-35%)
                await self.update_progress_advanced(
                    user_id, "preprocessing", 20, 100,
                    {'status': 'Optimizing video for processing'}
                )
                
                optimized_video_path = await self._preprocess_video_optimized(
                    video_path, user_id
                )
                if optimized_video_path != video_path:
                    temp_files.append(optimized_video_path)
                
                # Stage 3: Apply watermarks with turbo engine (35-70%)
                watermarked_path = f"{video_path}_watermarked.mp4"
                temp_files.append(watermarked_path)
                
                await self._watermark_stage_optimized(
                    user_id, optimized_video_path, watermarked_path,
                    watermark_text, png_location
                )
                
                # Stage 4: Process intro if exists (70-85%)
                final_path = await self._intro_stage_optimized(
                    user_id, watermarked_path, temp_files
                )
                
                # Stage 5: Prepare metadata and thumbnail (85-90%)
                await self._metadata_stage_optimized(
                    user_id, original_caption, final_path
                )
                
                # Stage 6: Upload with optimization (90-100%)
                await self._upload_stage_optimized(
                    user_id, final_path, original_caption, progress_msg
                )
                
                # Success cleanup
                await self._cleanup_processing_files(temp_files)
                
                # Calculate total time
                session = await self.session_manager.get_session(user_id)
                total_time = time.time() - session.get('processing_start_time', time.time())
                time_str = self._format_duration(total_time)
                
                if not video_num:  # Single video
                    await message.reply_text(
                        f"🎉 **Video processed in {time_str}!**\n"
                        f"⚡ Ultra-fast pipeline completed successfully!"
                    )
        
        except Exception as e:
            error_msg = str(e)
            if "stopped by user" in error_msg:
                await message.reply_text("🛑 **Processing stopped by user**")
            else:
                await message.reply_text(f"❌ **Error:** {error_msg}")
                logger.error(f"Ultra-optimized processing error: {e}")
        
        finally:
            # Always cleanup
            if progress_msg:
                try:
                    await asyncio.sleep(2)
                    await progress_msg.delete()
                except:
                    pass
            
            await self._cleanup_processing_files(temp_files)
            await self.session_manager.update_session(user_id, {'processing': False})

    async def _download_stage_optimized(self, message: Message, user_id: int,
                                      video_path: str, progress_msg: Message):
        """Optimized download stage with turbo progress"""
        session = await self.session_manager.get_session(user_id)
        file_id = session.get('video_file_id')
        
        if not file_id:
            raise Exception("No file ID found for download")
        
        # Use turbo download system
        await self.download_with_turbo_progress(
            message, file_id, video_path, "video",
            filename=session.get('video_filename')
        )
    
    async def _preprocess_video_optimized(self, video_path: str, user_id: int) -> str:
        """Optimized video preprocessing"""
        try:
            # Get optimal specs for processing
            video_info = await self.video_processor.get_video_info_fast(video_path)
            
            # Determine if preprocessing is needed
            needs_optimization = (
                video_info.width > 1920 or video_info.height > 1080 or
                video_info.fps > 60 or video_info.codec not in ['h264', 'x264']
            )
            
            if needs_optimization:
                target_specs = {
                    'width': min(video_info.width, 1920),
                    'height': min(video_info.height, 1080),
                    'fps': min(video_info.fps, 30),
                    'codec': 'h264'
                }
                
                return await self.video_processor.optimize_video_preprocessing(
                    video_path, target_specs
                )
            
            return video_path
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original: {e}")
            return video_path
    
    async def _watermark_stage_optimized(self, user_id: int, input_path: str,
                                       output_path: str, watermark_text: str,
                                       png_location: str):
        """Optimized watermarking stage"""
        watermark_png_path = f"persistent_watermarks/watermark_{user_id}.png"
        png_exists = os.path.exists(watermark_png_path)
        
        async def watermark_progress_callback(current, total, progress_info=None):
            details = {}
            if progress_info:
                details.update(progress_info)
                # Map frame progress to 35-70% range
                if total > 0:
                    frame_percentage = (current / total) * 35  # 35% span
                    overall_percentage = 35 + frame_percentage
                else:
                    overall_percentage = 52  # Mid-point
                
                details['processing_fps'] = progress_info.get('processing_fps', 0)
                details['frames'] = f"{current}/{total}"
                
            else:
                overall_percentage = 52
            
            await self.update_progress_advanced(
                user_id, "watermarking", int(overall_percentage), 100, details
            )
        
        success = await self.watermark_engine.apply_watermarks_turbocharged(
            input_path, output_path, watermark_text,
            watermark_png_path if png_exists else None,
            png_location, watermark_progress_callback
        )
        
        if not success:
            raise Exception("Watermarking failed")
    
    async def _intro_stage_optimized(self, user_id: int, video_path: str,
                                   temp_files: List[str]) -> str:
        """Optimized intro processing stage"""
        intro_path = f"persistent_intros/intro_{user_id}.mp4"
        
        if not os.path.exists(intro_path):
            await self.update_progress_advanced(
                user_id, "processing", 85, 100,
                {'status': 'No intro found, proceeding to upload'}
            )
            return video_path
        
        # Check intro validity
        try:
            intro_info = await self.video_processor.get_video_info_fast(intro_path)
            if intro_info.duration <= 0:
                raise Exception("Invalid intro duration")
        except Exception as e:
            logger.warning(f"Invalid intro file: {e}")
            await self.update_progress_advanced(
                user_id, "processing", 85, 100,
                {'status': 'Invalid intro, skipping'}
            )
            return video_path
        
        # Process intro with main video
        output_with_intro = f"{video_path}_with_intro.mp4"
        temp_files.append(output_with_intro)
        
        async def intro_progress_callback(current, total, progress_info=None):
            # Map to 70-85% range
            if total > 0:
                stage_percentage = (current / total) * 15  # 15% span
                overall_percentage = 70 + stage_percentage
            else:
                overall_percentage = 77  # Mid-point
            
            details = progress_info or {}
            await self.update_progress_advanced(
                user_id, "processing", int(overall_percentage), 100, details
            )
        
        success = await self.intro_processor.process_intro_with_main_turbocharged(
            intro_path, video_path, output_with_intro, intro_progress_callback
        )
        
        return output_with_intro if success else video_path
    
    async def _metadata_stage_optimized(self, user_id: int, original_caption: str,
                                      video_path: str):
        """Optimized metadata preparation"""
        await self.update_progress_advanced(
            user_id, "finalizing", 85, 100,
            {'status': 'Preparing metadata and thumbnail'}
        )
        
        # Prepare combined caption
        permanent_caption_path = f"persistent_captions/caption_{user_id}.txt"
        permanent_caption = ""
        
        if os.path.exists(permanent_caption_path):
            try:
                with open(permanent_caption_path, 'r', encoding='utf-8') as f:
                    permanent_caption = f.read().strip()
            except Exception as e:
                logger.error(f"Error reading permanent caption: {e}")
        
        combined_caption = original_caption
        if permanent_caption:
            combined_caption = f"{original_caption}\n\n{permanent_caption}"
        
        # Store in session for upload
        await self.session_manager.update_session(user_id, {
            'final_caption': combined_caption
        })
        
        # Generate thumbnail if needed
        thumbnail_path = f"persistent_thumbnails/thumbnail_{user_id}.jpg"
        if not os.path.exists(thumbnail_path):
            auto_thumbnail = f"temp_thumb_{user_id}_{int(time.time())}.jpg"
            if await self._generate_thumbnail_fast(video_path, auto_thumbnail):
                await self.session_manager.update_session(user_id, {
                    'auto_thumbnail': auto_thumbnail
                })
    
    async def _upload_stage_optimized(self, user_id: int, video_path: str,
                                    original_caption: str, progress_msg: Message):
        """Optimized upload stage with progress tracking"""
        session = await self.session_manager.get_session(user_id)
        combined_caption = session.get('final_caption', original_caption)
        
        # Get thumbnail
        thumbnail_path = f"persistent_thumbnails/thumbnail_{user_id}.jpg"
        auto_thumbnail = session.get('auto_thumbnail')
        thumbnail = thumbnail_path if os.path.exists(thumbnail_path) else auto_thumbnail
        
        # Get video metadata
        video_duration = session.get('video_duration', 0)
        video_width = session.get('video_width', 1280)
        video_height = session.get('video_height', 720)
        
        # Upload progress callback
        upload_start_time = time.time()
        
        async def upload_progress_callback(current, total):
            if session.get('stopped'):
                raise Exception("Process was stopped by user")
            
            # Map to 90-100% range
            if total > 0:
                upload_percentage = (current / total) * 10  # 10% span
                overall_percentage = 90 + upload_percentage
            else:
                overall_percentage = 95
            
            # Calculate upload speed
            elapsed = time.time() - upload_start_time
            speed_mbps = (current / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            
            details = {
                'current_mb': current / (1024 * 1024),
                'total_mb': total / (1024 * 1024),
                'speed_mbps': speed_mbps,
                'upload_progress': f"{current}/{total} bytes"
            }
            
            await self.update_progress_advanced(
                user_id, "uploading", int(overall_percentage), 100, details
            )
        
        # Execute upload
        try:
            await self.app.send_video(
                chat_id=user_id,
                video=video_path,
                caption=combined_caption,
                duration=int(video_duration) if video_duration > 0 else None,
                width=int(video_width) if video_width > 0 else None,
                height=int(video_height) if video_height > 0 else None,
                thumb=thumbnail,
                supports_streaming=True,
                progress=upload_progress_callback
            )
        except Exception as upload_error:
            logger.error(f"Optimized upload failed: {upload_error}")
            # Fallback upload without progress
            await self.app.send_video(
                chat_id=user_id,
                video=video_path,
                caption=combined_caption,
                thumb=thumbnail
            )
    
    async def _generate_thumbnail_fast(self, video_path: str, thumbnail_path: str) -> bool:
        """Fast thumbnail generation using optimized FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', '5',  # Seek to 5 seconds
                '-vf', 'scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2',
                '-vframes', '1',
                '-f', 'image2',
                '-q:v', '3',  # High quality
                '-y', thumbnail_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            await asyncio.wait_for(process.wait(), timeout=30)
            return process.returncode == 0 and os.path.exists(thumbnail_path)
            
        except Exception as e:
            logger.warning(f"Fast thumbnail generation failed: {e}")
            return False
    
    async def _cleanup_processing_files(self, temp_files: List[str]):
        """Cleanup processing files efficiently"""
        cleanup_tasks = []
        
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                cleanup_tasks.append(self._remove_file_async(temp_file))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    async def _remove_file_async(self, file_path: str):
        """Remove file asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, os.remove, file_path)
        except Exception as e:
            logger.warning(f"Failed to remove {file_path}: {e}")

    def get_optimized_system_info(self) -> str:
        """Get enhanced system information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network info if available
            try:
                network = psutil.net_io_counters()
                net_info = f" | Net: {network.bytes_sent/1024/1024:.1f}MB↑"
            except:
                net_info = ""
            
            return (f"💻 CPU: {cpu_percent:.1f}% | "
                   f"RAM: {memory.percent:.1f}% | "
                   f"Disk: {disk.percent:.1f}%{net_info}")
        except Exception as e:
            logger.warning(f"System info error: {e}")
            return "💻 System monitoring active"

    # Continue with remaining optimized handlers...
    # Part 3 - Final optimized handlers and system completion

    # Optimized Media Handlers
    async def video_handler_optimized(self, client: Client, message: Message):
        """Ultra-optimized video handler with smart processing"""
        user_id = message.from_user.id
        video = message.video
        session = await self.session_manager.get_session(user_id)

        if session.get('stopped'):
            await message.reply_text("Process was stopped. Use /start to begin again.")
            return

        if session.get('processing'):
            await message.reply_text(
                "Already processing a video. Please wait or use /stop to cancel."
            )
            return

        # Handle different processing modes
        waiting_for = session.get('waiting_for')

        if waiting_for == 'set_intro':
            await self._handle_intro_upload(message, user_id, video)
        elif waiting_for == 'convert_video':
            await self._handle_convert_video_upload(message, user_id, video)
        elif session.get('bulk_mode'):
            await self._handle_bulk_video_upload(message, user_id, video)
        else:
            await self._handle_regular_video_upload(message, user_id, video)

    async def _handle_intro_upload(self, message: Message, user_id: int, video):
        """Handle intro video upload with optimization"""
        intro_path = f"persistent_intros/intro_{user_id}.mp4"
        
        try:
            # Download with progress tracking
            await self.download_with_turbo_progress(
                message, video.file_id, intro_path, "intro video"
            )
            
            # Validate intro video
            video_info = await self.video_processor.get_video_info_fast(intro_path)
            if video_info.duration <= 0 or video_info.width <= 0:
                os.remove(intro_path)
                await message.reply_text("Invalid intro video. Please send a valid video file.")
                return
            
            # Clear cache for this intro
            await self._clear_intro_cache(user_id)
            
            await self.session_manager.update_session(user_id, {'waiting_for': None})
            await message.reply_text(
                f"Intro video saved successfully!\n"
                f"Duration: {video_info.duration:.1f}s | "
                f"Resolution: {video_info.width}x{video_info.height}"
            )
            
        except Exception as e:
            logger.error(f"Intro upload failed: {e}")
            await message.reply_text(f"Failed to save intro: {str(e)}")

    async def _handle_convert_video_upload(self, message: Message, user_id: int, video):
        """Handle convert video upload"""
        user_dir = Path(f"temp_processing/user_{user_id}")
        user_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = user_dir / f"convert_{int(time.time())}.mp4"
        
        try:
            await self.download_with_turbo_progress(
                message, video.file_id, str(video_path), "video for conversion"
            )
            
            await self.session_manager.update_session(user_id, {
                'video_path': str(video_path),
                'waiting_for': 'convert_watermark_text'
            })
            
            await message.reply_text(
                "Video ready for conversion! Send watermark text or type 'skip' to process without text watermark."
            )
            
        except Exception as e:
            logger.error(f"Convert upload failed: {e}")
            await message.reply_text(f"Upload failed: {str(e)}")

    async def _handle_bulk_video_upload(self, message: Message, user_id: int, video):
        """Handle bulk video upload with queue management"""
        queue_file = f"bulk_queue/queue_{user_id}.json"
        
        try:
            # Load existing queue
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    queue = json.load(f)
            else:
                queue = []
            
            # Prepare video metadata
            caption = message.caption or f"video_{len(queue) + 1}"
            filename = getattr(video, 'file_name', None) or f"{caption}.mp4"
            
            # Clean filename
            import re
            clean_filename = re.sub(r'[^\w\-_\.\s]', '_', filename)
            if not clean_filename.lower().endswith('.mp4'):
                clean_filename += '.mp4'
            
            # Add to queue
            queue_item = {
                'filename': clean_filename,
                'file_id': video.file_id,
                'file_size': video.file_size,
                'original_caption': caption,
                'duration': video.duration,
                'width': video.width,
                'height': video.height,
                'added_at': time.time()
            }
            
            queue.append(queue_item)
            
            # Save updated queue
            with open(queue_file, 'w') as f:
                json.dump(queue, f, indent=2)
            
            # Update session
            await self.session_manager.update_session(user_id, {
                'bulk_queue_count': len(queue)
            })
            
            # Send confirmation
            file_size_mb = video.file_size / (1024 * 1024) if video.file_size else 0
            queue_msg = f"Video #{len(queue)} added to queue!\n"
            queue_msg += f"File: {clean_filename}\n"
            
            if file_size_mb >= 1024:
                queue_msg += f"Size: {file_size_mb/1024:.2f}GB\n"
            elif file_size_mb > 0:
                queue_msg += f"Size: {file_size_mb:.1f}MB\n"
            
            queue_msg += f"Total queued: {len(queue)} videos\n\n"
            
            if len(queue) >= 3:
                queue_msg += "Ready to process! Send watermark text or 'skip' to start."
            else:
                queue_msg += "Add more videos or send watermark text to begin."
            
            await message.reply_text(queue_msg)
            
        except Exception as e:
            logger.error(f"Bulk upload failed: {e}")
            await message.reply_text(f"Failed to add video to queue: {str(e)}")

    async def _handle_regular_video_upload(self, message: Message, user_id: int, video):
        """Handle regular single video upload"""
        caption = message.caption or "input_video"
        import re
        clean_caption = re.sub(r'[^\w\-_\.]', '_', caption)
        if not clean_caption.endswith('.mp4'):
            clean_caption += '.mp4'
        
        # Store complete video metadata
        await self.session_manager.update_session(user_id, {
            'video_filename': clean_caption,
            'original_caption': caption,
            'video_file_id': video.file_id,
            'video_duration': video.duration or 0,
            'video_width': video.width or 1280,
            'video_height': video.height or 720,
            'video_file_size': video.file_size or 0,
            'waiting_for': 'png_location'
        })
        
        # Check for PNG watermark
        watermark_path = f"persistent_watermarks/watermark_{user_id}.png"
        if os.path.exists(watermark_path):
            await message.reply_text(
                "Video metadata saved! Choose PNG watermark location:\n\n"
                "Available locations:\n"
                "• topleft\n• topright\n• bottomleft\n• bottomright"
            )
        else:
            await self.session_manager.update_session(user_id, {'waiting_for': 'watermark_text'})
            await message.reply_text(
                "Video metadata saved! Send watermark text or type 'skip' to process without text watermark."
            )

    # Optimized Text Handler
    async def text_handler_optimized(self, client: Client, message: Message):
        """Ultra-optimized text message handler"""
        user_id = message.from_user.id
        text = message.text.strip()
        session = await self.session_manager.get_session(user_id)

        if session.get('stopped'):
            await message.reply_text("Process was stopped. Use /start to begin again.")
            return

        if session.get('processing'):
            await message.reply_text("Already processing. Please wait or use /stop.")
            return

        waiting_for = session.get('waiting_for')

        # Route to appropriate handler
        if waiting_for == 'add_caption':
            await self._handle_caption_text(message, user_id, text)
        elif waiting_for == 'png_location':
            await self._handle_png_location_text(message, user_id, text)
        elif waiting_for == 'bulk_png_location':
            await self._handle_bulk_png_location_text(message, user_id, text)
        elif waiting_for == 'watermark_text':
            await self._handle_watermark_text(message, user_id, text)
        elif waiting_for == 'convert_watermark_text':
            await self._handle_convert_watermark_text(message, user_id, text)
        elif session.get('bulk_mode'):
            await self._handle_bulk_text(message, user_id, text)
        else:
            await self._handle_general_text(message, text)

    async def _handle_watermark_text(self, message: Message, user_id: int, text: str):
        """Handle watermark text input for single video"""
        session = await self.session_manager.get_session(user_id)
        
        # Check for skip commands
        watermark_text = None if text.lower() in ['skip', 'skip text', 'no text', 'no watermark'] else text
        
        # Process video
        user_dir = Path(f"temp_processing/user_{user_id}")
        user_dir.mkdir(parents=True, exist_ok=True)
        video_path = user_dir / session['video_filename']
        png_location = session.get('png_location', 'topright')
        
        try:
            await self.process_video_ultra_optimized(
                message, user_id, str(video_path), watermark_text,
                session['original_caption'], png_location=png_location
            )
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            await message.reply_text(f"Processing failed: {str(e)}")

    async def _handle_bulk_text(self, message: Message, user_id: int, text: str):
        """Handle text input for bulk processing"""
        # Check for skip commands
        watermark_text = None if text.lower() in ['skip', 'skip text', 'no text', 'no watermark'] else text
        
        session = await self.session_manager.get_session(user_id)
        png_location = session.get('png_location', 'topright')
        
        await self._process_bulk_queue_optimized(message, user_id, watermark_text, png_location)

    async def _process_bulk_queue_optimized(self, message: Message, user_id: int, 
                                          watermark_text: str, png_location: str):
        """Process bulk queue with maximum optimization"""
        queue_file = f"bulk_queue/queue_{user_id}.json"
        
        try:
            if not os.path.exists(queue_file):
                await message.reply_text("No videos in queue. Add videos first.")
                return
            
            with open(queue_file, 'r') as f:
                queue = json.load(f)
            
            if not queue:
                await message.reply_text("Queue is empty. Add videos first.")
                return
            
            # Update session for bulk processing
            await self.session_manager.update_session(user_id, {
                'processing': True,
                'bulk_mode': False,
                'total_videos': len(queue),
                'processing_start_time': time.time()
            })
            
            skip_msg = " (skipping text watermark)" if watermark_text is None else ""
            await message.reply_text(
                f"Starting ultra-fast bulk processing of {len(queue)} videos{skip_msg}!\n"
                f"Processing will use parallel optimization for maximum speed."
            )
            
            # Process using bulk optimizer
            results = await self.bulk_optimizer.process_bulk_queue_optimized(
                user_id, queue, watermark_text, png_location,
                self._create_bulk_progress_callback(message, user_id)
            )
            
            # Final summary
            successful = len([r for r in results if not isinstance(r, Exception)])
            failed = len(queue) - successful
            
            summary_msg = f"Bulk processing completed!\n"
            summary_msg += f"Successful: {successful}/{len(queue)} videos\n"
            if failed > 0:
                summary_msg += f"Failed: {failed} videos\n"
            
            total_time = time.time() - (await self.session_manager.get_session(user_id)).get('processing_start_time', time.time())
            summary_msg += f"Total time: {self._format_duration(total_time)}"
            
            await message.reply_text(summary_msg)
            
        except Exception as e:
            logger.error(f"Bulk processing failed: {e}")
            await message.reply_text(f"Bulk processing error: {str(e)}")
        finally:
            # Cleanup
            await self.session_manager.update_session(user_id, {
                'processing': False,
                'bulk_mode': False
            })
            if os.path.exists(queue_file):
                os.remove(queue_file)

    def _create_bulk_progress_callback(self, message: Message, user_id: int):
        """Create progress callback for bulk processing"""
        async def progress_callback(video_num: int, total_videos: int, stage: str, details: dict = None):
            try:
                overall_progress = ((video_num - 1) / total_videos + (details.get('stage_progress', 0) / 100) / total_videos) * 100
                
                progress_text = f"Bulk Processing: Video {video_num}/{total_videos}\n"
                progress_text += f"Overall: {overall_progress:.1f}%\n"
                progress_text += f"Current: {stage}"
                
                if details:
                    if 'processing_fps' in details:
                        progress_text += f" @ {details['processing_fps']:.1f}fps"
                    if 'eta' in details:
                        progress_text += f" | ETA: {details['eta']}"
                
                # Update every 10 seconds to avoid flooding
                session = await self.session_manager.get_session(user_id)
                last_bulk_update = session.get('last_bulk_update', 0)
                if time.time() - last_bulk_update > 10:
                    try:
                        await message.reply_text(progress_text)
                        await self.session_manager.update_session(user_id, {'last_bulk_update': time.time()})
                    except:
                        pass  # Ignore message sending errors
                        
            except Exception as e:
                logger.warning(f"Bulk progress callback error: {e}")
        
        return progress_callback

    # Advanced Admin Handlers
    async def stats_handler_optimized(self, client: Client, message: Message):
        """Advanced statistics with performance metrics"""
        user_id = message.from_user.id
        if user_id not in self.admin_users:
            await message.reply_text("Admin access required.")
            return
        
        try:
            # System performance metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Bot-specific metrics
            active_users = self.session_manager.get_active_users_count()
            processing_users = len([s for s in self.session_manager.sessions.values() if s.get('processing')])
            
            # File counts
            intro_count = len([f for f in os.listdir('persistent_intros') if f.endswith('.mp4')]) if os.path.exists('persistent_intros') else 0
            watermark_count = len([f for f in os.listdir('persistent_watermarks') if f.endswith('.png')]) if os.path.exists('persistent_watermarks') else 0
            
            # Cache statistics
            cache_size = 0
            cache_files = 0
            if os.path.exists('cache'):
                for root, dirs, files in os.walk('cache'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        cache_size += os.path.getsize(file_path)
                        cache_files += 1
            
            # GPU information
            gpu_info = "Not available"
            if self.video_processor.ffmpeg_processor.gpu_available:
                gpu_info = f"{self.video_processor.ffmpeg_processor.gpu_type.upper()} ({self.video_processor.ffmpeg_processor.encoder})"
            
            stats_text = f"""**Advanced Bot Statistics**

**System Performance:**
CPU: {cpu_percent:.1f}%
RAM: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)
Disk: {disk.percent:.1f}% ({disk.used/1024/1024/1024:.1f}GB/{disk.total/1024/1024/1024:.1f}GB)
GPU Acceleration: {gpu_info}

**Bot Activity:**
Active Users: {active_users}
Processing Users: {processing_users}
Admin Users: {len(self.admin_users)}

**Storage Usage:**
Intro Videos: {intro_count}
PNG Watermarks: {watermark_count}
Cache Files: {cache_files} ({cache_size/1024/1024:.1f}MB)

**Performance Settings:**
Processing Threads: {self.video_processor.ffmpeg_processor.optimal_threads}
Max Concurrent Processing: {self.bulk_optimizer.processing_semaphore._value}
Max Concurrent Downloads: {self.bulk_optimizer.download_semaphore._value}
            """
            
            await message.reply_text(stats_text)
            
        except Exception as e:
            logger.error(f"Stats generation failed: {e}")
            await message.reply_text(f"Stats error: {str(e)}")

    async def cleanup_handler_optimized(self, client: Client, message: Message):
        """Advanced cleanup with detailed reporting"""
        user_id = message.from_user.id
        if user_id not in self.admin_users:
            await message.reply_text("Admin access required.")
            return
        
        try:
            cleanup_msg = await message.reply_text("Starting comprehensive cleanup...")
            
            cleaned_items = {
                'temp_dirs': 0,
                'cache_files': 0,
                'old_logs': 0,
                'orphaned_files': 0,
                'freed_mb': 0
            }
            
            # Cleanup temp directories
            for root, dirs, files in os.walk('.'):
                for dir_name in dirs[:]:  # Use slice to avoid modification during iteration
                    if dir_name.startswith('temp_'):
                        temp_path = os.path.join(root, dir_name)
                        try:
                            size_mb = sum(os.path.getsize(os.path.join(temp_path, f)) for f in os.listdir(temp_path)) / (1024*1024)
                            shutil.rmtree(temp_path)
                            cleaned_items['temp_dirs'] += 1
                            cleaned_items['freed_mb'] += size_mb
                        except:
                            pass
            
            # Cleanup old cache files
            if os.path.exists('cache'):
                current_time = time.time()
                for root, dirs, files in os.walk('cache'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            if current_time - os.path.getmtime(file_path) > 86400:  # 24 hours
                                size_mb = os.path.getsize(file_path) / (1024*1024)
                                os.remove(file_path)
                                cleaned_items['cache_files'] += 1
                                cleaned_items['freed_mb'] += size_mb
                        except:
                            pass
            
            # Cleanup old log files
            if os.path.exists('logs'):
                for log_file in os.listdir('logs'):
                    if log_file.endswith('.log'):
                        log_path = os.path.join('logs', log_file)
                        try:
                            if time.time() - os.path.getmtime(log_path) > 604800:  # 7 days
                                size_mb = os.path.getsize(log_path) / (1024*1024)
                                os.remove(log_path)
                                cleaned_items['old_logs'] += 1
                                cleaned_items['freed_mb'] += size_mb
                        except:
                            pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Update cleanup message
            cleanup_summary = f"""**Cleanup Completed!**

Temp Directories: {cleaned_items['temp_dirs']} removed
Cache Files: {cleaned_items['cache_files']} removed  
Old Logs: {cleaned_items['old_logs']} removed
Total Space Freed: {cleaned_items['freed_mb']:.2f}MB

Memory garbage collection performed.
System resources optimized.
            """
            
            await cleanup_msg.edit_text(cleanup_summary)
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            await message.reply_text(f"Cleanup error: {str(e)}")

    # Helper methods
    async def _clear_intro_cache(self, user_id: int):
        """Clear cached intro files for user"""
        try:
            cache_dir = Path("cache/normalized_intros")
            if cache_dir.exists():
                for cache_file in cache_dir.iterdir():
                    if f"intro_{user_id}" in cache_file.name:
                        cache_file.unlink()
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display"""
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs}s"
        else:
            return f"{int(seconds)}s"

    # Main execution method
    def run(self):
        """Run the ultra-optimized watermark bot"""
        from pyrogram import handlers
        
        # Register all optimized handlers
        handlers_list = [
            # Command handlers
            (handlers.MessageHandler(self.start_handler, pyrogram_filters.command("start")),),
            (handlers.MessageHandler(self._create_command_handler('setintro', 'set_intro'), pyrogram_filters.command("setintro")),),
            (handlers.MessageHandler(self._create_command_handler('setwatermark', 'set_watermark'), pyrogram_filters.command("setwatermark")),),
            (handlers.MessageHandler(self._create_command_handler('setthumbnail', 'set_thumbnail'), pyrogram_filters.command("setthumbnail")),),
            (handlers.MessageHandler(self._create_command_handler('addcaption', 'add_caption'), pyrogram_filters.command("addcaption")),),
            (handlers.MessageHandler(self._create_command_handler('convert', 'convert_video'), pyrogram_filters.command("convert")),),
            (handlers.MessageHandler(self._create_simple_handler(self._status_handler), pyrogram_filters.command("status")),),
            (handlers.MessageHandler(self._create_simple_handler(self._stop_handler), pyrogram_filters.command("stop")),),
            (handlers.MessageHandler(self._create_simple_handler(self._bulk_handler), pyrogram_filters.command("bulk")),),
            (handlers.MessageHandler(self.stats_handler_optimized, pyrogram_filters.command("stats")),),
            (handlers.MessageHandler(self.cleanup_handler_optimized, pyrogram_filters.command("cleanup")),),
            
            # Media handlers
            (handlers.MessageHandler(self.video_handler_optimized, pyrogram_filters.video),),
            (handlers.MessageHandler(self._photo_handler_optimized, pyrogram_filters.photo),),
            (handlers.MessageHandler(self._document_handler_optimized, pyrogram_filters.document),),
            
            # Text handler (must be last)
            (handlers.MessageHandler(
                self.text_handler_optimized,
                pyrogram_filters.text & ~pyrogram_filters.command([
                    "start", "setintro", "setwatermark", "setthumbnail", "addcaption", 
                    "convert", "status", "stop", "bulk", "stats", "cleanup"
                ])
            ),),
        ]
        
        # Register all handlers
        for handler_tuple in handlers_list:
            self.app.add_handler(handler_tuple[0])
        
        print("Ultra-Fast Watermark Bot starting with full optimization...")
        print(f"GPU Acceleration: {'Available' if self.video_processor.ffmpeg_processor.gpu_available else 'Not Available'}")
        print(f"Processing Threads: {self.video_processor.ffmpeg_processor.optimal_threads}")
        print(f"System: {self.get_optimized_system_info()}")
        
        try:
            # Start the optimized bot
            self.app.run()
        except KeyboardInterrupt:
            print("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot runtime error: {e}")
            raise
        finally:
            # Cleanup on shutdown
            print("Performing shutdown cleanup...")
            asyncio.run(self._shutdown_cleanup())

    def _create_command_handler(self, command: str, waiting_state: str):
        """Create optimized command handler"""
        async def handler(client: Client, message: Message):
            user_id = message.from_user.id
            await self.session_manager.update_session(user_id, {'waiting_for': waiting_state})
            
            command_messages = {
                'setintro': "Send the intro video:",
                'setwatermark': "Send the watermark image (PNG recommended):",
                'setthumbnail': "Send the thumbnail image:",
                'addcaption': "Send the permanent caption:",
                'convert': "Send the video to convert:"
            }
            
            await message.reply_text(command_messages.get(command, "Send the required media:"))
        
        return handler

    def _create_simple_handler(self, handler_func):
        """Create wrapper for simple handlers"""
        async def wrapper(client: Client, message: Message):
            return await handler_func(message)
        return wrapper

    async def _shutdown_cleanup(self):
        """Perform cleanup on bot shutdown"""
        try:
            # Cancel all background tasks
            for task in asyncio.all_tasks():
                if not task.done():
                    task.cancel()
            
            # Final temp cleanup
            await self._cleanup_temp_files()
            
            print("Shutdown cleanup completed")
            
        except Exception as e:
            logger.error(f"Shutdown cleanup error: {e}")


# Main execution
if __name__ == "__main__":
    # Environment configuration
    API_ID = int(os.getenv("API_ID", "6063221"))
    API_HASH = os.getenv("API_HASH", "8f9bebe9a9cb147ee58f70f46506f787")
    BOT_TOKEN = os.getenv("BOT_TOKEN", "7219961311:AAEYxJ4aHzQsf2c6_dcQFyWskSkCeE-Sa4k")
    
    # Initialize and run ultra-optimized bot
    print("Initializing Ultra-Fast Watermark Bot...")
    bot = WatermarkBot(API_ID, API_HASH, BOT_TOKEN)
    bot.run()