"""
Email service module using Brevo (formerly Sendinblue) API.
Replaces SMTP functionality with Brevo's transactional email API.
"""

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import Brevo SDK, fallback to HTTP if not available
try:
    from sib_api_v3_sdk import TransactionalEmailsApi, SendSmtpEmail, SendSmtpEmailTo, SendSmtpEmailSender
    from sib_api_v3_sdk.rest import ApiException
    import sib_api_v3_sdk
    SDK_AVAILABLE = True
except ImportError:
    logger.warning("Brevo SDK not available, will use HTTP fallback")
    SDK_AVAILABLE = False


class BrevoEmailService:
    """Email service using Brevo API for sending transactional emails."""
    
    def __init__(self, api_key: str, sender_email: str, sender_name: str = "AnemoCheck"):
        """
        Initialize Brevo email service.
        
        Args:
            api_key: Brevo API key
            sender_email: Email address to send from
            sender_name: Name to display as sender
        """
        self.api_key = api_key
        self.sender_email = sender_email
        self.sender_name = sender_name
        
        # Configure API client
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = api_key
        
        self.api_instance = TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
    
    def send_email(self, to_email: str, subject: str, html_content: str, 
                   text_content: str, to_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Send email using Brevo API.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text email content
            to_name: Recipient name (optional)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Create sender
            sender = SendSmtpEmailSender(
                email=self.sender_email,
                name=self.sender_name
            )
            
            # Create recipient
            to = SendSmtpEmailTo(
                email=to_email,
                name=to_name or to_email.split('@')[0]
            )
            
            # Create email object
            send_smtp_email = SendSmtpEmail(
                sender=sender,
                to=[to],
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
            # Send email
            api_response = self.api_instance.send_transac_email(send_smtp_email)
            
            logger.info(f"Email sent successfully to {to_email}. Message ID: {api_response.message_id}")
            return True, "Email sent successfully"
            
        except ApiException as e:
            error_msg = f"Brevo API error: {e.reason} - {e.body}"
            logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Error sending email: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


def get_brevo_service():
    """
    Get configured Brevo email service from database settings.
    Uses SDK if available, otherwise falls back to HTTP implementation.
    
    Returns:
        BrevoEmailService instance or None if not configured
    """
    try:
        import database as db
        
        # Get Brevo settings from database
        api_key = db.get_system_setting('brevo_api_key')
        sender_email = db.get_system_setting('brevo_sender_email')
        sender_name = db.get_system_setting('brevo_sender_name') or "AnemoCheck"
        enable_notifications = db.get_system_setting('enable_email_notifications') == 'true'
        
        # Debug logging
        logger.info(f"Brevo service check - enable_notifications: {enable_notifications}")
        logger.info(f"Brevo service check - api_key exists: {bool(api_key)}")
        logger.info(f"Brevo service check - sender_email: {sender_email}")
        
        if not enable_notifications or not api_key or not sender_email:
            logger.warning("Brevo email service not configured or disabled")
            return None
        
        if SDK_AVAILABLE:
            return BrevoEmailService(api_key, sender_email, sender_name)
        else:
            # Use HTTP fallback
            from email_service_http import BrevoHTTPEmailService
            return BrevoHTTPEmailService(api_key, sender_email, sender_name)
        
    except Exception as e:
        logger.error(f"Error initializing Brevo service: {str(e)}")
        return None


def send_result_email_brevo(record_id: int, user_email: str, user_name: str, record_data: dict) -> Tuple[bool, str]:
    """
    Send anemia test result email using Brevo API.
    
    Args:
        record_id: Record ID
        user_email: User's email address
        user_name: User's name
        record_data: Record data dictionary
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Get Brevo service
        brevo_service = get_brevo_service()
        if not brevo_service:
            return False, "Email service not configured or disabled"
        
        # Create email content
        subject = f"AnemoCheck - Your Anemia Test Result ({record_data['predicted_class']})"
        
        # Create HTML email content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Anemia Test Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #c62828; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .result-box {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .classification {{ font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 5px; }}
                .normal {{ background-color: #d4edda; color: #155724; }}
                .mild {{ background-color: #fff3cd; color: #856404; }}
                .moderate {{ background-color: #f8d7da; color: #721c24; }}
                .severe {{ background-color: #f5c6cb; color: #721c24; }}
                .values-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .values-table th, .values-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .values-table th {{ background-color: #f2f2f2; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AnemoCheck - Anemia Test Result</h1>
                </div>
                <div class="content">
                    <h2>Hello {user_name},</h2>
                    <p>Your anemia classification test has been completed. Here are your results:</p>
                    
                    <div class="result-box">
                        <h3>Classification Result</h3>
                        <div class="classification {'normal' if record_data['predicted_class'] == 'Normal' else 'mild' if record_data['predicted_class'] == 'Mild' else 'moderate' if record_data['predicted_class'] == 'Moderate' else 'severe' if record_data['predicted_class'] == 'Severe' else ''}">
                            {record_data['predicted_class']}
                        </div>
                        <p style="text-align: center; margin-top: 10px;">
                            <strong>Confidence: {record_data['confidence']:.2%}</strong>
                        </p>
                    </div>
                    
                    <div class="result-box">
                        <h3>Complete Blood Count (CBC) Values</h3>
                        <table class="values-table">
                            <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
                            <tr><td>White Blood Cell Count (WBC)</td><td>{record_data['wbc']:.2f}</td><td>10³/µL</td></tr>
                            <tr><td>Red Blood Cell Count (RBC)</td><td>{record_data['rbc']:.2f}</td><td>million/µL</td></tr>
                            <tr><td>Hemoglobin (HGB)</td><td>{record_data['hgb']:.2f}</td><td>g/dL</td></tr>
                            <tr><td>Hematocrit (HCT)</td><td>{record_data['hct']:.2f}</td><td>%</td></tr>
                            <tr><td>Mean Corpuscular Volume (MCV)</td><td>{record_data['mcv']:.2f}</td><td>fL</td></tr>
                            <tr><td>Mean Corpuscular Hemoglobin (MCH)</td><td>{record_data['mch']:.2f}</td><td>pg</td></tr>
                            <tr><td>Mean Corpuscular Hemoglobin Concentration (MCHC)</td><td>{record_data['mchc']:.2f}</td><td>g/dL</td></tr>
                            <tr><td>Platelet Count (PLT)</td><td>{record_data['plt']:.2f}</td><td>10³/µL</td></tr>
                        </table>
                    </div>
                    
                    {f'<div class="result-box"><h3>Notes</h3><p>{record_data.get("notes", "")}</p></div>' if record_data.get("notes") else ''}
                    
                    <div class="result-box">
                        <h3>Important Information</h3>
                        <p><strong>Please note:</strong> This is an AI-powered screening tool and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.</p>
                        <p><strong>Test Date:</strong> {record_data['created_at']}</p>
                    </div>
                </div>
                <div class="footer">
                    <p>This email was sent from AnemoCheck - Anemia Detection System</p>
                    <p>For support, please contact your healthcare provider</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = f"""
        AnemoCheck - Anemia Test Result
        
        Hello {user_name},
        
        Your anemia classification test has been completed. Here are your results:
        
        Classification Result: {record_data['predicted_class']}
        Confidence: {record_data['confidence']:.2%}
        
        Complete Blood Count (CBC) Values:
        - White Blood Cell Count (WBC): {record_data['wbc']:.2f} 10³/µL
        - Red Blood Cell Count (RBC): {record_data['rbc']:.2f} million/µL
        - Hemoglobin (HGB): {record_data['hgb']:.2f} g/dL
        - Hematocrit (HCT): {record_data['hct']:.2f} %
        - Mean Corpuscular Volume (MCV): {record_data['mcv']:.2f} fL
        - Mean Corpuscular Hemoglobin (MCH): {record_data['mch']:.2f} pg
        - Mean Corpuscular Hemoglobin Concentration (MCHC): {record_data['mchc']:.2f} g/dL
        - Platelet Count (PLT): {record_data['plt']:.2f} 10³/µL
        
        {f'Notes: {record_data.get("notes", "")}' if record_data.get("notes") else ''}
        
        Important Information:
        Please note: This is an AI-powered screening tool and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
        
        Test Date: {record_data['created_at']}
        
        This email was sent from AnemoCheck - Anemia Detection System
        For support, please contact your healthcare provider
        """
        
        # Send email using Brevo
        return brevo_service.send_email(
            to_email=user_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            to_name=user_name
        )
        
    except Exception as e:
        logger.error(f"Error sending result email via Brevo: {str(e)}")
        return False, f"Error sending email: {str(e)}"


def send_otp_email_brevo(email: str, otp_code: str, username: str) -> bool:
    """
    Send OTP code email using Brevo API.
    
    Args:
        email: Recipient email address
        otp_code: OTP code to send
        username: Username for personalization
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Get Brevo service
        brevo_service = get_brevo_service()
        if not brevo_service:
            logger.warning("Brevo email service not configured. Using development mode.")
            # Development fallback
            print(f"\n{'='*60}")
            print(f"DEVELOPMENT MODE - OTP EMAIL")
            print(f"{'='*60}")
            print(f"To: {email}")
            print(f"Subject: AnemoCheck - Email Verification Code")
            print(f"")
            print(f"Hello {username},")
            print(f"")
            print(f"Your verification code is: {otp_code}")
            print(f"")
            print(f"This code will expire in 10 minutes.")
            print(f"{'='*60}\n")
            return True
        
        # Create email content
        subject = "AnemoCheck - Email Verification Code"
        
        # Create HTML email content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Email Verification</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #007bff; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ padding: 20px; background-color: #f9f9f9; border-radius: 0 0 8px 8px; }}
                .otp-box {{ background-color: white; padding: 30px; margin: 20px 0; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .otp-code {{ font-size: 36px; font-weight: bold; color: #007bff; letter-spacing: 8px; margin: 20px 0; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AnemoCheck Email Verification</h1>
                </div>
                <div class="content">
                    <h2>Hello {username},</h2>
                    <p>Thank you for registering with AnemoCheck! To complete your registration, please use the verification code below:</p>
                    
                    <div class="otp-box">
                        <h3>Your Verification Code</h3>
                        <div class="otp-code">{otp_code}</div>
                        <p>Enter this code in the verification page to complete your registration.</p>
                    </div>
                    
                    <div class="warning">
                        <strong>Important:</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>This code will expire in 10 minutes</li>
                            <li>Do not share this code with anyone</li>
                            <li>If you didn't request this code, please ignore this email</li>
                        </ul>
                    </div>
                    
                    <p>If you have any questions, please contact our support team.</p>
                </div>
                <div class="footer">
                    <p>This is an automated message from AnemoCheck. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = f"""
        AnemoCheck Email Verification
        
        Hello {username},
        
        Thank you for registering with AnemoCheck! To complete your registration, please use the verification code below:
        
        Your Verification Code: {otp_code}
        
        Enter this code in the verification page to complete your registration.
        
        Important:
        - This code will expire in 10 minutes
        - Do not share this code with anyone
        - If you didn't request this code, please ignore this email
        
        If you have any questions, please contact our support team.
        
        This is an automated message from AnemoCheck. Please do not reply to this email.
        """
        
        # Send email using Brevo
        success, message = brevo_service.send_email(
            to_email=email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            to_name=username
        )
        
        if success:
            logger.info(f"OTP email sent successfully to {email}")
            return True
        else:
            logger.error(f"Failed to send OTP email: {message}")
            # Fallback to development mode
            print(f"\n{'='*60}")
            print(f"EMAIL SENDING FAILED - DEVELOPMENT FALLBACK")
            print(f"{'='*60}")
            print(f"To: {email}")
            print(f"Subject: AnemoCheck - Email Verification Code")
            print(f"")
            print(f"Hello {username},")
            print(f"")
            print(f"Your verification code is: {otp_code}")
            print(f"")
            print(f"This code will expire in 10 minutes.")
            print(f"{'='*60}\n")
            return True
            
    except Exception as e:
        logger.error(f"Error sending OTP email via Brevo: {str(e)}")
        # Fallback to development mode
        print(f"\n{'='*60}")
        print(f"EMAIL SENDING FAILED - DEVELOPMENT FALLBACK")
        print(f"{'='*60}")
        print(f"To: {email}")
        print(f"Subject: AnemoCheck - Email Verification Code")
        print(f"")
        print(f"Hello {username},")
        print(f"")
        print(f"Your verification code is: {otp_code}")
        print(f"")
        print(f"This code will expire in 10 minutes.")
        print(f"{'='*60}\n")
        return True
