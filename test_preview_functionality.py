"""
Test script for preview downsampling and streaming functionality.
"""

import asyncio
import numpy as np
import logging
from app.services.preview_downsampler import PreviewDownsampler, DownsampleMethod
from app.services.preview_streamer import PreviewStreamer
from app.services.preview_performance import performance_monitor, preview_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_preview_downsampling():
    """Test preview downsampling functionality."""
    logger.info("Testing preview downsampling...")
    
    # Create test mask
    mask = np.random.randint(0, 2, size=(128, 128, 128), dtype=np.uint8)
    
    # Test different downsampling methods
    downsampler = PreviewDownsampler(target_size=(64, 64, 64))
    
    for method in DownsampleMethod:
        logger.info(f"Testing {method.value} downsampling...")
        downsampled, scale_factors = downsampler.downsample_mask(mask, method=method)
        
        logger.info(f"Original shape: {mask.shape}")
        logger.info(f"Downsampled shape: {downsampled.shape}")
        logger.info(f"Scale factors: {scale_factors}")
        
        # Test multi-resolution previews
        previews = downsampler.create_multi_resolution_previews(mask, scales=[1.0, 0.5, 0.25])
        logger.info(f"Generated {len(previews)} multi-resolution previews")
        
        # Test visual quality optimization
        optimized = downsampler.optimize_for_visual_quality(downsampled)
        logger.info(f"Optimized shape: {optimized.shape}")


async def test_preview_streaming():
    """Test preview streaming functionality."""
    logger.info("Testing preview streaming...")
    
    # Create test mask
    mask = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.uint8)
    
    # Create streamer
    streamer = PreviewStreamer(
        job_id="test_job",
        frequency=2.0,  # 2 Hz
        target_size=(32, 32, 32)
    )
    
    # Start streaming
    await streamer.start_streaming(mask)
    
    # Simulate mask updates
    for i in range(5):
        # Update mask slightly
        new_mask = mask.copy()
        new_mask[i*10:(i+1)*10, :, :] = 1
        streamer.update_mask(new_mask)
        
        # Wait for preview to be sent
        await asyncio.sleep(0.6)  # Slightly longer than 1/frequency
    
    # Stop streaming
    await streamer.stop_streaming()
    logger.info("Preview streaming test completed")


async def test_performance_monitoring():
    """Test performance monitoring functionality."""
    logger.info("Testing performance monitoring...")
    
    # Clear previous metrics
    performance_monitor.metrics_history.clear()
    preview_cache.clear()
    
    # Simulate some operations
    for i in range(10):
        timing_key = performance_monitor.start_timing("test_job", "test_operation")
        await asyncio.sleep(0.01)  # Simulate work
        duration = performance_monitor.stop_timing(timing_key)
        logger.info(f"Operation {i}: {duration:.2f}ms")
    
    # Get performance stats
    stats = performance_monitor.get_job_performance("test_job")
    logger.info(f"Job performance stats: {stats}")
    
    # Test cache
    test_data = np.random.rand(32, 32, 32)
    preview_cache.put("test_key", test_data)
    cached_data = preview_cache.get("test_key")
    
    if cached_data is not None:
        logger.info("Cache test passed")
    else:
        logger.error("Cache test failed")
    
    cache_stats = preview_cache.get_stats()
    logger.info(f"Cache stats: {cache_stats}")


async def main():
    """Run all tests."""
    logger.info("Starting preview functionality tests...")
    
    try:
        await test_preview_downsampling()
        await test_performance_monitoring()
        await test_preview_streaming()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
