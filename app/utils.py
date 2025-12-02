import logging
import sys
import os

# Configure logging with detailed format
def configure_logging():
    """Configure logging for the application"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, 'app.log'))
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = configure_logging()

def log_prediction(input_data: dict, prediction: str, confidence: float, probabilities: list) -> None:
    """Log prediction results"""
    logger.info(
        f"Prediction - Input: {input_data} | Result: {prediction} | "
        f"Confidence: {confidence:.2f}% | Probs: [No: {probabilities[0]*100:.2f}%, Yes: {probabilities[1]*100:.2f}%]"
    )

def log_error(error_msg: str, exc_info=None) -> None:
    """Log errors with optional exception info"""
    logger.error(f"Error: {error_msg}", exc_info=exc_info)

def log_info(msg: str) -> None:
    """Log info messages"""
    logger.info(msg)

def log_debug(msg: str) -> None:
    """Log debug messages"""
    logger.debug(msg)