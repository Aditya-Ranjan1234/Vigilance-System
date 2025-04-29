"""
Notifier module for sending alerts to various destinations.

This module provides functionality to send alerts via different channels
such as email, SMS, and push notifications.
"""

import os
import time
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, Any, List, Optional
from pathlib import Path
import cv2
import numpy as np

from vigilance_system.utils.logger import get_logger
from vigilance_system.utils.config import config

# Initialize logger
logger = get_logger(__name__)


class AlertNotifier:
    """
    Sends alerts through various notification channels.
    
    Supports email, SMS, and other notification methods based on configuration.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Singleton pattern implementation to ensure only one notifier exists.
        
        Returns:
            AlertNotifier: The singleton AlertNotifier instance
        """
        if cls._instance is None:
            cls._instance = super(AlertNotifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the alert notifier."""
        if self._initialized:
            return
            
        # Load configuration
        self.notification_config = config.get('alerts.notification', {})
        
        # Initialize notification channels
        self.email_enabled = self.notification_config.get('email', {}).get('enabled', False)
        self.sms_enabled = self.notification_config.get('sms', {}).get('enabled', False)
        
        # Create alert history directory
        self.alert_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'alerts'
        )
        os.makedirs(self.alert_dir, exist_ok=True)
        
        # Initialize alert history
        self.alert_history = []
        self.max_history_size = 100
        
        self._initialized = True
        logger.info(f"Alert notifier initialized with email={self.email_enabled}, SMS={self.sms_enabled}")
    
    def send_alert(self, alert: Dict[str, Any], frame: Optional[np.ndarray] = None) -> bool:
        """
        Send an alert through configured notification channels.
        
        Args:
            alert: Alert information dictionary
            frame: Optional frame image associated with the alert
        
        Returns:
            bool: True if alert was sent successfully through at least one channel
        """
        # Add alert to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)
        
        # Save alert to file
        self._save_alert(alert, frame)
        
        # Send through configured channels
        success = False
        
        if self.email_enabled:
            email_success = self._send_email_alert(alert, frame)
            success = success or email_success
        
        if self.sms_enabled:
            sms_success = self._send_sms_alert(alert)
            success = success or sms_success
        
        return success
    
    def _save_alert(self, alert: Dict[str, Any], frame: Optional[np.ndarray] = None) -> None:
        """
        Save alert information and image to disk.
        
        Args:
            alert: Alert information dictionary
            frame: Optional frame image associated with the alert
        """
        try:
            # Create timestamp-based filename
            timestamp = int(time.time())
            alert_type = alert.get('type', 'unknown')
            camera = alert.get('camera', 'unknown')
            
            # Save alert data as JSON
            alert_filename = f"{timestamp}_{camera}_{alert_type}.json"
            alert_path = os.path.join(self.alert_dir, alert_filename)
            
            with open(alert_path, 'w') as f:
                json.dump(alert, f, indent=2)
            
            # Save frame image if provided
            if frame is not None:
                image_filename = f"{timestamp}_{camera}_{alert_type}.jpg"
                image_path = os.path.join(self.alert_dir, image_filename)
                cv2.imwrite(image_path, frame)
                
            logger.info(f"Saved alert to {alert_path}")
            
        except Exception as e:
            logger.error(f"Error saving alert: {str(e)}")
    
    def _send_email_alert(self, alert: Dict[str, Any], frame: Optional[np.ndarray] = None) -> bool:
        """
        Send an alert via email.
        
        Args:
            alert: Alert information dictionary
            frame: Optional frame image associated with the alert
        
        Returns:
            bool: True if email was sent successfully
        """
        try:
            # Get email configuration
            email_config = self.notification_config.get('email', {})
            recipients = email_config.get('recipients', [])
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            smtp_username = email_config.get('smtp_username')
            smtp_password = email_config.get('smtp_password')
            
            if not recipients or not smtp_server or not smtp_username or not smtp_password:
                logger.error("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = f"Security Alert: {alert.get('type', 'Unknown').title()}"
            msg['From'] = smtp_username
            msg['To'] = ', '.join(recipients)
            
            # Add alert message
            alert_message = alert.get('message', 'Security alert detected')
            msg.attach(MIMEText(f"{alert_message}\n\nTimestamp: {time.ctime(alert.get('timestamp', time.time()))}", 'plain'))
            
            # Add image if provided
            if frame is not None:
                # Convert frame to JPEG
                _, img_data = cv2.imencode('.jpg', frame)
                img = MIMEImage(img_data.tobytes())
                img.add_header('Content-Disposition', 'attachment', filename='alert.jpg')
                msg.attach(img)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Sent email alert to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def _send_sms_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Send an alert via SMS.
        
        Args:
            alert: Alert information dictionary
        
        Returns:
            bool: True if SMS was sent successfully
        """
        try:
            # Get SMS configuration
            sms_config = self.notification_config.get('sms', {})
            phone_numbers = sms_config.get('phone_numbers', [])
            service = sms_config.get('service', '').lower()
            
            if not phone_numbers or not service:
                logger.error("SMS configuration incomplete")
                return False
            
            # Create message
            alert_message = alert.get('message', 'Security alert detected')
            
            # Send via configured service
            if service == 'twilio':
                return self._send_twilio_sms(phone_numbers, alert_message, sms_config)
            elif service == 'aws_sns':
                return self._send_aws_sns(phone_numbers, alert_message, sms_config)
            else:
                logger.error(f"Unsupported SMS service: {service}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending SMS alert: {str(e)}")
            return False
    
    def _send_twilio_sms(self, phone_numbers: List[str], message: str, sms_config: Dict[str, Any]) -> bool:
        """
        Send SMS via Twilio.
        
        Args:
            phone_numbers: List of recipient phone numbers
            message: SMS message
            sms_config: SMS configuration dictionary
        
        Returns:
            bool: True if SMS was sent successfully
        """
        try:
            # Import Twilio client (only when needed)
            from twilio.rest import Client
            
            # Get Twilio configuration
            account_sid = sms_config.get('twilio_account_sid')
            auth_token = sms_config.get('twilio_auth_token')
            from_number = sms_config.get('twilio_from_number')
            
            if not account_sid or not auth_token or not from_number:
                logger.error("Twilio configuration incomplete")
                return False
            
            # Initialize Twilio client
            client = Client(account_sid, auth_token)
            
            # Send to each recipient
            for phone_number in phone_numbers:
                client.messages.create(
                    body=message,
                    from_=from_number,
                    to=phone_number
                )
            
            logger.info(f"Sent Twilio SMS alert to {len(phone_numbers)} recipients")
            return True
            
        except ImportError:
            logger.error("Twilio package not installed")
            return False
        except Exception as e:
            logger.error(f"Error sending Twilio SMS: {str(e)}")
            return False
    
    def _send_aws_sns(self, phone_numbers: List[str], message: str, sms_config: Dict[str, Any]) -> bool:
        """
        Send SMS via AWS SNS.
        
        Args:
            phone_numbers: List of recipient phone numbers
            message: SMS message
            sms_config: SMS configuration dictionary
        
        Returns:
            bool: True if SMS was sent successfully
        """
        try:
            # Import boto3 (only when needed)
            import boto3
            
            # Initialize SNS client
            sns = boto3.client('sns')
            
            # Send to each recipient
            for phone_number in phone_numbers:
                sns.publish(
                    PhoneNumber=phone_number,
                    Message=message
                )
            
            logger.info(f"Sent AWS SNS alert to {len(phone_numbers)} recipients")
            return True
            
        except ImportError:
            logger.error("boto3 package not installed")
            return False
        except Exception as e:
            logger.error(f"Error sending AWS SNS: {str(e)}")
            return False
    
    def get_alert_history(self) -> List[Dict[str, Any]]:
        """
        Get the alert history.
        
        Returns:
            List[Dict[str, Any]]: List of recent alerts
        """
        return self.alert_history.copy()
    
    def clear_alert_history(self) -> None:
        """Clear the alert history."""
        self.alert_history = []
        logger.info("Cleared alert history")


# Create a default instance
notifier = AlertNotifier()
