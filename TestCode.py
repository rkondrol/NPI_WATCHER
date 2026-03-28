import PyUber
import pandas as pd
import urllib.parse
import getpass
from typing import Optional, Dict, Any, List, Union
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
import sys
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import socket

# Custom logging configuration with your specified path
log_file_path = r"\\f21p-NAS-FabScheduler.f21prod.mfg.intel.com\DMOIESystems\mdt\jobs\NPI_WATCHER\NPI_WATCHER_ESCALATION_LOG.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, mode='a')
    ]
)
logger = logging.getLogger(__name__)

class F11xDB:
    """Enhanced F11x Database Manager with transaction support"""
    
    def __init__(self):
        self.engine = None
        self.current_user = getpass.getuser()
        logger.info(f"Initializing F11xDB for user: {self.current_user}")
        self._connect_with_fallback()

    def _connect_with_fallback(self):
        """Try multiple ODBC drivers with transaction-optimized settings"""
        drivers_to_try = [
            'ODBC Driver 18 for SQL Server',
            'ODBC Driver 17 for SQL Server',
            'ODBC Driver 13 for SQL Server',
            'ODBC Driver 11 for SQL Server',
            'SQL Server Native Client 11.0',
            'SQL Server Native Client 10.0',
            'SQL Server'
        ]
        
        server = 'sql2604-or1-in.amr.corp.intel.com,3181'
        database = 'f11xprod'
        uid = 'f11xprod_so'
        pwd = 'fG1Tx7gKiVr7y0d'
        
        logger.info(f"Attempting to connect to {server}/{database}")
        
        for driver in drivers_to_try:
            try:
                logger.debug(f"Trying driver: {driver}")
                params = urllib.parse.quote_plus(
                    f'Driver={{{driver}}};'
                    f'Server={server};'
                    f'Database={database};'
                    f'UID={uid};'
                    f'PWD={pwd};'
                    'TrustServerCertificate=yes;'
                    'Encrypt=no;'
                    'Connection Timeout=30;'
                    'Command Timeout=300;'
                    'Mars_Connection=no;'
                )
                
                self.engine = create_engine(
                    "mssql+pyodbc:///?odbc_connect=%s" % params,
                    use_setinputsizes=False,
                    fast_executemany=True,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    pool_size=3,
                    max_overflow=5,
                    connect_args={
                        "timeout": 30,
                        "autocommit": True
                    }
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1 as test"))
                    test_value = result.fetchone()[0]
                    if test_value != 1:
                        raise Exception("Connection test failed")
                
                logger.info(f"Successfully connected using driver: {driver}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to connect with driver '{driver}': {str(e)}")
                if self.engine:
                    try:
                        self.engine.dispose()
                    except:
                        pass
                    self.engine = None
                continue
        
        error_msg = "Could not establish database connection with any available driver"
        logger.error(error_msg)
        raise Exception(error_msg)

    def pull(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SELECT query and return DataFrame"""
        try:
            logger.debug(f"Executing SQL query with {len(params) if params else 0} parameters")
            if params:
                return pd.read_sql(text(sql), self.engine, params=params)
            else:
                return pd.read_sql(sql, self.engine)
        except Exception as e:
            logger.error(f"Error executing pull query: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            self.engine.dispose()

class IEIndicatorsDB:
    """Database connection for IEIndicators database"""
    
    def __init__(self):
        self.engine = None
        self.current_user = getpass.getuser()
        logger.info(f"Initializing IEIndicatorsDB for user: {self.current_user}")
        self._connect_with_fallback()

    def _connect_with_fallback(self):
        """Connect to IEIndicators database"""
        drivers_to_try = [
            'ODBC Driver 18 for SQL Server',
            'ODBC Driver 17 for SQL Server',
            'ODBC Driver 13 for SQL Server',
            'ODBC Driver 11 for SQL Server',
            'SQL Server Native Client 11.0',
            'SQL Server Native Client 10.0',
            'SQL Server'
        ]
        
        server = 'nmiesql104.amr.corp.intel.com\\sql01,1788'
        database = 'IEIndicators'
        
        logger.info(f"Attempting to connect to {server}/{database}")
        
        for driver in drivers_to_try:
            try:
                logger.debug(f"Trying driver: {driver}")
                params = urllib.parse.quote_plus(
                    f'Driver={{{driver}}};'
                    f'Server={server};'
                    f'Database={database};'
                    'Trusted_Connection=yes;'
                    'TrustServerCertificate=yes;'
                    'Connection Timeout=30;'
                    'Command Timeout=300;'
                )
                
                self.engine = create_engine(
                    "mssql+pyodbc:///?odbc_connect=%s" % params,
                    use_setinputsizes=False,
                    fast_executemany=True,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    pool_size=3,
                    max_overflow=5,
                    connect_args={
                        "timeout": 30,
                        "autocommit": True
                    }
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                logger.info(f"Successfully connected using driver: {driver}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to connect with driver '{driver}': {str(e)}")
                continue
        
        error_msg = "Could not establish IEIndicators database connection with any available driver"
        logger.error(error_msg)
        raise Exception(error_msg)

    def pull(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SELECT query and return DataFrame"""
        try:
            logger.debug(f"Executing IEIndicators SQL query with {len(params) if params else 0} parameters")
            if params:
                return pd.read_sql(text(sql), self.engine, params=params)
            else:
                return pd.read_sql(sql, self.engine)
        except Exception as e:
            logger.error(f"Error executing IEIndicators pull query: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            self.engine.dispose()

class EmailService:
    """Simplified email service based on working CQT Defender implementation"""
    
    def __init__(self, test_mode=False):
        # Your SMTP credentials (same as before)
        self.sender_email = "NMDMOPROD@intel.com"
        self.smtp_username = "AMR/grp_nmdmoprod"
        self.smtp_password = "@ProductionNMDMO2026"
        
        self.bcc_recipient = "rishitha.kondrolla@intel.com"  # Fixed BCC recipient
        self.test_mode = test_mode
        
        # Test mode specific recipients
        self.test_mode_recipients = [
            "rishitha.kondrolla@intel.com",
            "uriel.mendiola@intel.com"
        ]
        
        # Use only the working configuration from CQT Defender
        self.smtp_config = {
            "server": "smtpauth.intel.com", 
            "port": 587, 
            "use_tls": True, 
            "auth": True, 
            "use_ssl": False
        }
        
        logger.info(f"EmailService initialized - Test Mode: {test_mode}")
        logger.info(f"Sender email: {self.sender_email}")
        logger.info(f"SMTP username: {self.smtp_username}")
        logger.info(f"Using proven SMTP config: {self.smtp_config}")
        logger.info(f"Fixed BCC recipient: {self.bcc_recipient}")
        if test_mode:
            logger.info(f"Test mode recipients: {self.test_mode_recipients}")
    
    def quick_smtp_test(self):
        """Simplified port connectivity test"""
        logger.info("Testing SMTP port connectivity...")
        
        try:
            sock = socket.create_connection(("smtpauth.intel.com", 587), timeout=10)
            sock.close()
            logger.info("SUCCESS: Port 587 is reachable on smtpauth.intel.com")
            return [("smtpauth.intel.com", 587)]
        except Exception as e:
            logger.warning(f"Port connectivity test failed: {e}")
            return [("smtpauth.intel.com", 587)]  # Return it anyway
    
    def test_smtp_connection(self):
        """Simplified SMTP connection test based on working CQT Defender approach"""
        
        # Use the same simple config that works in CQT Defender
        working_config = {
            "server": "smtpauth.intel.com", 
            "port": 587, 
            "use_tls": True, 
            "auth": True, 
            "use_ssl": False
        }
        
        try:
            logger.info(f"Testing SMTP: {working_config['server']}:{working_config['port']}")
            
            # Use the exact same approach as CQT Defender
            server = smtplib.SMTP(working_config['server'], working_config['port'], timeout=30)
            
            with server:
                server.starttls()
                
                # Try the same authentication approach as CQT Defender
                try:
                    server.login(self.smtp_username, self.smtp_password)
                    logger.info("Authentication successful with primary username")
                except smtplib.SMTPAuthenticationError:
                    # Same fallback as CQT Defender
                    alt_username = self.smtp_username.split('/')[-1]
                    logger.info(f"Retrying with username: {alt_username}")
                    server.login(alt_username, self.smtp_password)
                    logger.info("Authentication successful with fallback username")
                
                server.noop()
            
            logger.info("SMTP connection test successful")
            return working_config
            
        except Exception as e:
            logger.error(f"SMTP test failed: {e}")
            return working_config  # Return it anyway, might work during actual send
    
    def send_escalation_email(self, recipient_email, subject, body, recipient_name, state, cc_emails=None, department=None):
        """Send escalation email using simplified approach"""
        
        # Prepare CC list from ALL ACTIVE_EMAILS
        cc_list = []
        if cc_emails:
            if isinstance(cc_emails, str):
                cc_list = [cc_emails]
            elif isinstance(cc_emails, list):
                cc_list = cc_emails
        
        if self.test_mode:
            # TEST MODE: Send to test recipients
            logger.info(f"TEST MODE: Sending actual test email")
            logger.info(f"  Original TO: {recipient_name} ({recipient_email}) for {state}")
            logger.info(f"  Original CC: {cc_list}")
            logger.info(f"  TEST MODE: Redirecting to test recipients")
            
            test_subject = f"[TEST MODE] {subject}"
            test_body = f"""[TEST MODE - ESCALATION SIMULATION]

ORIGINAL ESCALATION DETAILS:
- TO: {recipient_name} ({recipient_email})
- STATE: {state}
- DEPARTMENT: {department}
- ORIGINAL CC: {cc_list}

ORIGINAL MESSAGE:
{body}

--- END TEST MODE MESSAGE ---"""
            
            return self._send_actual_email(
                recipient_email="rishitha.kondrolla@intel.com",
                subject=test_subject,
                body=test_body,
                cc_emails=["uriel.mendiola@intel.com"],
                bcc_email=None
            )
        
        # Production mode - send actual email
        return self._send_actual_email(
            recipient_email=recipient_email,
            subject=subject,
            body=body,
            cc_emails=cc_list,
            bcc_email=self.bcc_recipient
        )
    
    def _send_actual_email(self, recipient_email, subject, body, cc_emails=None, bcc_email=None):
        """Send email using CQT Defender's proven approach"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Add CC recipients
            if cc_emails:
                if isinstance(cc_emails, list):
                    msg['Cc'] = ', '.join(cc_emails)
                else:
                    msg['Cc'] = cc_emails
            
            # Create HTML version
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    {body.replace(chr(10), '<br>')}
                </body>
            </html>
            """
            
            # Create plain text version as fallback
            text_body = body
            
            # Attach both parts
            text_part = MIMEText(text_body, 'plain')
            html_part = MIMEText(html_body, 'html')
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Prepare all recipients: TO + CC + BCC
            all_recipients = [recipient_email]
            if cc_emails:
                if isinstance(cc_emails, list):
                    all_recipients.extend(cc_emails)
                else:
                    all_recipients.append(cc_emails)
            if bcc_email:
                all_recipients.append(bcc_email)
            
            # USE THE EXACT SAME SMTP APPROACH AS CQT DEFENDER
            with smtplib.SMTP('smtpauth.intel.com', 587, timeout=600) as smtp_server:
                smtp_server.starttls()
                try:
                    smtp_server.login(self.smtp_username, self.smtp_password)
                    logger.debug("Authentication successful with primary username")
                except smtplib.SMTPAuthenticationError:
                    # Same simple fallback as CQT Defender
                    alt_username = self.smtp_username.split('/')[-1]
                    logger.info(f"Retrying with username: {alt_username}")
                    smtp_server.login(alt_username, self.smtp_password)
                    logger.debug("Authentication successful with fallback username")
                
                # Send the message
                smtp_server.sendmail(self.sender_email, all_recipients, msg.as_string())
            
            mode = "TEST MODE" if self.test_mode else "PRODUCTION"
            logger.info(f"{mode} escalation email sent successfully:")
            logger.info(f"  FROM: {self.sender_email}")
            logger.info(f"  TO: {recipient_email}")
            logger.info(f"  CC: {cc_emails}")
            if bcc_email:
                logger.info(f"  BCC: {bcc_email}")
            
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication failed: {e}")
            logger.error(f"Username: {self.smtp_username}")
            return False
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"SMTP Recipients refused: {e}")
            logger.error(f"Recipients: {all_recipients}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

class XeusDataProcessor:
    """Main class for processing Xeus data with F_CALENDAR shift detection"""
    
    def __init__(self, test_mode=False):
        self.f11x_db = None
        self.xeus_conn = None
        self.test_mode = test_mode
        self.email_service = EmailService(test_mode=test_mode)
        
        if test_mode:
            logger.info("XEUS DATA PROCESSOR INITIALIZED IN TEST MODE")

    def get_rw_departments(self) -> pd.DataFrame:
        """Get ALL departments from RW_Department table for ACTIVE_EMAILS"""
        try:
            logger.info("Fetching ALL departments from F11x for ACTIVE_EMAILS")
            with F11xDB() as db:
                sql = "SELECT RW_Department, ACTIVE_EMAILS FROM RW_Department WHERE Active = 1"
                df = db.pull(sql)
                logger.info(f"Successfully pulled {len(df)} active departments")
                
                # Log departments with ACTIVE_EMAILS
                if 'ACTIVE_EMAILS' in df.columns:
                    dept_with_emails = df[df['ACTIVE_EMAILS'].notna()]
                    logger.info(f"Found {len(dept_with_emails)} departments with ACTIVE_EMAILS")
                    
                    # Show what ACTIVE_EMAILS we found
                    for _, row in dept_with_emails.iterrows():
                        emails_preview = str(row['ACTIVE_EMAILS'])[:100] + "..." if len(str(row['ACTIVE_EMAILS'])) > 100 else str(row['ACTIVE_EMAILS'])
                        logger.info(f"  {row['RW_Department']}: {emails_preview}")
                else:
                    logger.warning("ACTIVE_EMAILS column not found in RW_Department table")
                
                return df
        except Exception as e:
            logger.error(f"Error pulling RW_Department data: {e}")
            return pd.DataFrame()

    def get_all_active_emails(self, dept_df: pd.DataFrame) -> List[str]:
        """
        Simple approach - get ALL ACTIVE_EMAILS from ALL departments
        No mapping, no matching - just collect all ACTIVE_EMAILS
        """
        if dept_df.empty:
            return []
        
        if 'ACTIVE_EMAILS' not in dept_df.columns:
            logger.warning("ACTIVE_EMAILS column not found in RW_Department table")
            return []
        
        all_emails = []
        
        try:
            # Go through ALL departments and collect their ACTIVE_EMAILS
            for _, row in dept_df.iterrows():
                active_emails = row['ACTIVE_EMAILS']
                
                if pd.notna(active_emails) and str(active_emails).strip():
                    # Parse emails from this department
                    email_str = str(active_emails).strip()
                    emails = re.split(r'[;,\s\n\r]+', email_str)
                    
                    # Clean and add to list
                    for email in emails:
                        email = email.strip()
                        if email and '@' in email and '.' in email:
                            if email not in all_emails:  # Avoid duplicates
                                all_emails.append(email)
            
            logger.info(f"Collected {len(all_emails)} unique ACTIVE_EMAILS from all departments: {all_emails}")
            return all_emails
            
        except Exception as e:
            logger.error(f"Error collecting ACTIVE_EMAILS: {e}")
            return []

    def execute_xeus_query(self, dept_filter: str) -> pd.DataFrame:
        """Execute the main Xeus query"""
        query = f"""
        SELECT LOT,
               OPERATION,
               PROCESS_OPERATION,
               AREA,
               MODULE,
               HOURS_AT_OPERATION,
               CASE 
                   WHEN HOURS_AT_OPERATION < 2 THEN 'State 1'
                   WHEN HOURS_AT_OPERATION >= 2 AND HOURS_AT_OPERATION < 8 THEN 'State 2'
                   WHEN HOURS_AT_OPERATION >= 8 AND HOURS_AT_OPERATION < 12 THEN 'State 3'
                   WHEN HOURS_AT_OPERATION >= 12 THEN 'State 4'
                   ELSE 'Unknown'
               END AS STATE,
               CASE 
                   WHEN UPPER(AREA) IN ('METRO', 'ANALYTICAL') THEN 'Y'
                   ELSE 'N'
               END AS ISMETRO,
               GROUP_NAME,
               COMMENTS,
               LAST_EDITED
        FROM (
            SELECT P_F_LOT.LOT,
                   P_F_LOT.OPERATION,
                   P_SPC_LOT.PROCESS_OPERATION,
                   P_F_OPERATION_RUN_CARD.AREA,
                   P_F_OPERATION_RUN_CARD.MODULE,
                   ROUND ((SYSDATE - P_F_LOT.PREV_OPER_OUT_DATE) * 24, 2) AS HOURS_AT_OPERATION,
                   F_RW_GROUP.GROUP_NAME,
                   P_F_RW_COMMENTS_SUM.COMMENTS,
                   P_F_RW_COMMENTS_SUM.LAST_EDITED,
                   ROW_NUMBER() OVER (PARTITION BY P_F_LOT.LOT ORDER BY P_F_RW_COMMENTS_SUM.LAST_EDITED DESC) AS rn
            FROM (
                    (
                       (
                          (
                             (
                                F21_PROD_XEUSS.P_F_RW_LOT_GROUP P_F_RW_LOT_GROUP
                                INNER JOIN F21_PROD_XEUSS.F_RW_GROUP F_RW_GROUP
                                   ON (P_F_RW_LOT_GROUP.GROUP_ID = F_RW_GROUP.GROUP_ID))
                             INNER JOIN F21_PROD_XEUSS.P_F_LOT P_F_LOT
                                ON (P_F_LOT.LOT = P_F_RW_LOT_GROUP.LOT))
                          INNER JOIN F21_PROD_XEUSS.P_F_OPERATION_RUN_CARD P_F_OPERATION_RUN_CARD
                             ON (P_F_OPERATION_RUN_CARD.OPERATION = P_F_LOT.OPERATION))
                       INNER JOIN F21_PROD_XEUSS.F_RW_DEPT F_RW_DEPT
                          ON (F_RW_GROUP.DEPT_ID = F_RW_DEPT.DEPT_ID))
                    INNER JOIN F21S.P_SPC_LOT P_SPC_LOT
                       ON (P_F_LOT.LOT = P_SPC_LOT.LOT))
                 LEFT JOIN F21_PROD_XEUSS.P_F_RW_COMMENTS_SUM P_F_RW_COMMENTS_SUM
                    ON (P_F_LOT.LOT = CASE 
                                        WHEN INSTR(P_F_RW_COMMENTS_SUM.FLOW_UID, '!') > 0 
                                        THEN SUBSTR(P_F_RW_COMMENTS_SUM.FLOW_UID, 1, INSTR(P_F_RW_COMMENTS_SUM.FLOW_UID, '!') - 1)
                                        ELSE P_F_RW_COMMENTS_SUM.FLOW_UID
                                      END)
            WHERE {dept_filter}
              AND P_F_OPERATION_RUN_CARD.WW_END_TIME >= SYSDATE - 7
        ) ranked_comments
        WHERE rn = 1 OR rn IS NULL
        """
        
        try:
            logger.info("Connecting to Xeus database")
            conn = PyUber.connect(datasource='F21_PROD_XEUS')
            
            logger.info("Executing Xeus query...")
            cursor = conn.cursor()
            cursor.execute(query)
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            
            logger.info(f"Query completed: {len(df)} rows, {len(columns)} columns")
            logger.info(f"Columns returned: {columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error executing Xeus query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
            logger.info("Xeus connection closed")

    def get_current_shift_from_calendar(self) -> Dict[str, Any]:
        """
        Get current shift information from F_CALENDAR table in Xeus
        Returns shift details including SHIFT ID and timing
        """
        query = """
        SELECT 
            SYSDATE AS CURRENT_SYSTEM_TIME,
            FC.START_DATE,
            FC.END_DATE,
            FC.SHIFT,
            CASE 
                WHEN SYSDATE BETWEEN FC.START_DATE AND FC.END_DATE THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END AS SHIFT_STATUS
        FROM F_CALENDAR FC
        WHERE SYSDATE BETWEEN FC.START_DATE AND FC.END_DATE
        ORDER BY FC.START_DATE
        """
        
        try:
            logger.info("Fetching current shift from F_CALENDAR table in Xeus")
            
            conn = PyUber.connect(datasource='F21_PROD_XEUS')
            cursor = conn.cursor()
            cursor.execute(query)
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            if not results:
                logger.warning("No active shift found in F_CALENDAR for current time")
                return {
                    'success': False,
                    'message': 'No active shift found',
                    'current_shift': None,
                    'shift_details': None
                }
            
            # Convert to DataFrame for easier handling
            shift_df = pd.DataFrame(results, columns=columns)
            
            logger.info(f"F_CALENDAR query completed: {len(shift_df)} active shift(s) found")
            
            # Log all active shifts
            for _, row in shift_df.iterrows():
                logger.info(f"Active Shift: {row['SHIFT']}")
                logger.info(f"  Start: {row['START_DATE']}")
                logger.info(f"  End: {row['END_DATE']}")
                logger.info(f"  Status: {row['SHIFT_STATUS']}")
            
            # If multiple shifts are active, take the first one (or implement priority logic)
            current_shift_row = shift_df.iloc[0]
            
            shift_info = {
                'success': True,
                'current_shift': str(current_shift_row['SHIFT']),
                'current_system_time': current_shift_row['CURRENT_SYSTEM_TIME'],
                'start_date': current_shift_row['START_DATE'],
                'end_date': current_shift_row['END_DATE'],
                'shift_status': current_shift_row['SHIFT_STATUS'],
                'shift_details': shift_df.to_dict('records'),
                'total_active_shifts': len(shift_df)
            }
            
            logger.info(f"Current shift determined from F_CALENDAR: {shift_info['current_shift']}")
            logger.info(f"Shift period: {shift_info['start_date']} to {shift_info['end_date']}")
            
            return shift_info
            
        except Exception as e:
            logger.error(f"Error executing F_CALENDAR shift query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'message': str(e),
                'current_shift': None,
                'shift_details': None
            }
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
            logger.info("F_CALENDAR shift query connection closed")

    def map_calendar_shift_to_sgl(self, calendar_shift: str) -> str:
        """
        Map F_CALENDAR SHIFT value to SGL ID
        Simple mapping: SHIFT 4 -> SGL4, SHIFT 5 -> SGL5, etc.
        """
        # Convert to string and clean up
        shift_str = str(calendar_shift).strip()
        
        logger.info(f"Mapping calendar shift '{shift_str}' to SGL ID")
        
        # Extract number from shift value
        numbers = re.findall(r'\d+', shift_str)
        
        if numbers:
            shift_num = numbers[0]
            if shift_num in ['4', '5', '6', '7']:
                sgl_id = f'SGL{shift_num}'
                logger.info(f"Mapped calendar shift '{shift_str}' to '{sgl_id}'")
                return sgl_id
            else:
                logger.warning(f"Shift number '{shift_num}' not in expected range [4,5,6,7]")
        
        # Try direct mapping for common formats
        direct_mappings = {
            '4': 'SGL4', '5': 'SGL5', '6': 'SGL6', '7': 'SGL7',
            'SH4': 'SGL4', 'SH5': 'SGL5', 'SH6': 'SGL6', 'SH7': 'SGL7',
            'SHIFT4': 'SGL4', 'SHIFT5': 'SGL5', 'SHIFT6': 'SGL6', 'SHIFT7': 'SGL7',
            'SHIFT_4': 'SGL4', 'SHIFT_5': 'SGL5', 'SHIFT_6': 'SGL6', 'SHIFT_7': 'SGL7'
        }
        
        # Try uppercase version
        shift_upper = shift_str.upper()
        if shift_upper in direct_mappings:
            sgl_id = direct_mappings[shift_upper]
            logger.info(f"Direct mapped calendar shift '{shift_str}' to '{sgl_id}'")
            return sgl_id
        
        # Fallback
        logger.warning(f"Could not map calendar shift '{shift_str}' to SGL ID. Using SGL4 as default.")
        logger.warning(f"Available mappings: {list(direct_mappings.keys())}")
        return 'SGL4'

    def get_current_shift_info(self) -> Dict[str, Any]:
        """
        Get comprehensive current shift information using F_CALENDAR
        """
        # Get shift from calendar
        calendar_info = self.get_current_shift_from_calendar()
        
        if not calendar_info['success']:
            logger.error("Failed to get current shift from F_CALENDAR")
            return {
                'success': False,
                'message': calendar_info['message'],
                'current_shift': 'SGL4',  # Fallback
                'sgl_id': 'SGL4',
                'calendar_shift': None,
                'shift_description': 'Fallback - Calendar lookup failed'
            }
        
        calendar_shift = calendar_info['current_shift']
        sgl_id = self.map_calendar_shift_to_sgl(calendar_shift)
        
        # Create comprehensive shift info
        shift_info = {
            'success': True,
            'current_shift': sgl_id,  # SGL4, SGL5, SGL6, or SGL7
            'sgl_id': sgl_id,
            'calendar_shift': calendar_shift,  # Original F_CALENDAR.SHIFT value
            'current_system_time': calendar_info['current_system_time'],
            'start_date': calendar_info['start_date'],
            'end_date': calendar_info['end_date'],
            'shift_status': calendar_info['shift_status'],
            'total_active_shifts': calendar_info['total_active_shifts'],
            'shift_details': calendar_info['shift_details']
        }
        
        # Add shift description
        shift_descriptions = {
            'SGL4': f'Shift 4 Group Leader (Calendar: {calendar_shift})',
            'SGL5': f'Shift 5 Group Leader (Calendar: {calendar_shift})', 
            'SGL6': f'Shift 6 Group Leader (Calendar: {calendar_shift})',
            'SGL7': f'Shift 7 Group Leader (Calendar: {calendar_shift})'
        }
        
        shift_info['shift_description'] = shift_descriptions.get(sgl_id, f'Unknown Shift (Calendar: {calendar_shift})')
        
        logger.info(f"Current shift info from F_CALENDAR:")
        logger.info(f"  Calendar Shift: {calendar_shift}")
        logger.info(f"  Mapped to SGL: {sgl_id}")
        logger.info(f"  Description: {shift_info['shift_description']}")
        logger.info(f"  Active Period: {shift_info['start_date']} to {shift_info['end_date']}")
        
        return shift_info

    def get_ceid_mapping(self, target_operations: List[str]) -> pd.DataFrame:
        """
        Get CEID mapping for target operations from Xeus database
        """
        if not target_operations:
            logger.warning("No target operations provided for CEID mapping")
            return pd.DataFrame()
        
        # Remove None values and convert to string
        clean_operations = [str(op) for op in target_operations if op is not None and str(op) != 'nan']
        
        if not clean_operations:
            logger.warning("No valid target operations after cleaning")
            return pd.DataFrame()
        
        # Create placeholders for the IN clause
        placeholders = ','.join([f"'{op}'" for op in clean_operations])
        
        query = f"""
        SELECT DISTINCT
               P_F_OPERATION_RUN_CARD.CEID_LIST, 
               P_F_OPERATION_RUN_CARD.OPERATION
        FROM F21_PROD_XEUSS.P_F_OPERATION_RUN_CARD P_F_OPERATION_RUN_CARD
        WHERE (P_F_OPERATION_RUN_CARD.CEID_LIST IS NOT NULL)
          AND (P_F_OPERATION_RUN_CARD.OPERATION IN ({placeholders}))
          AND (P_F_OPERATION_RUN_CARD.WW_END_TIME >= SYSDATE - 7)
          AND (LENGTH(P_F_OPERATION_RUN_CARD.CEID_LIST) = 5)
        """
        
        try:
            logger.info(f"Fetching CEID mapping for {len(clean_operations)} target operations")
            logger.debug(f"Target operations: {clean_operations[:10]}...")  # Log first 10 for debugging
            
            conn = PyUber.connect(datasource='F21_PROD_XEUS')
            cursor = conn.cursor()
            cursor.execute(query)
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            ceid_df = pd.DataFrame(results, columns=columns)
            
            logger.info(f"CEID mapping query completed: {len(ceid_df)} mappings found")
            
            if not ceid_df.empty:
                # Log some statistics
                unique_operations = ceid_df['OPERATION'].nunique()
                unique_ceids = ceid_df['CEID_LIST'].nunique()
                logger.info(f"Found CEIDs for {unique_operations} operations, {unique_ceids} unique CEIDs")
                
                # Log sample mappings
                sample_mappings = ceid_df.head(5)
                logger.debug("Sample CEID mappings:")
                for _, row in sample_mappings.iterrows():
                    logger.debug(f"  {row['OPERATION']} -> {row['CEID_LIST']}")
            else:
                logger.warning("No CEID mappings found for the provided target operations")
            
            return ceid_df
            
        except Exception as e:
            logger.error(f"Error executing CEID mapping query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
            logger.info("CEID mapping query connection closed")

    def get_faceid_mapping_contacts(self, ceids: List[str]) -> pd.DataFrame:
        """
        Get all contacts for CEIDs from FACEIDMappingF11X_MES300 table in IEIndicators
        Returns shift group leaders (SGL4, SGL5, SGL6, SGL7) and managers
        """
        if not ceids:
            logger.warning("No CEIDs provided for FACEIDMapping lookup")
            return pd.DataFrame()
        
        clean_ceids = list(set([str(ceid) for ceid in ceids if ceid is not None and str(ceid) != 'nan']))
        
        if not clean_ceids:
            logger.warning("No valid CEIDs after cleaning")
            return pd.DataFrame()
        
        # Create placeholders for the IN clause
        placeholders = ','.join([f"'{ceid}'" for ceid in clean_ceids])
        
        query = f"""
        SELECT 
            CEID,
            FISTname,
            FactoryOrg,
            MgrGroup1Label,
            MgrGroup1,
            MgrGroup2Label, 
            MgrGroup2,
            MgrGroup3Label,
            MgrGroup3,
            SGL4,
            SGL5, 
            SGL6,
            SGL7,
            MgrGroup1WWID,
            MgrGroup3WWID,
            SGL4WWID,
            SGL5WWID,
            SGL6WWID,
            SGL7WWID,
            IEContact
        FROM FACEIDMappingF11X_MES300
        WHERE CEID IN ({placeholders})
        """
        
        try:
            logger.info(f"Fetching FACEIDMapping contacts for {len(clean_ceids)} CEIDs from IEIndicators database")
            logger.debug(f"CEIDs: {clean_ceids[:10]}...")
            
            with IEIndicatorsDB() as db:
                faceid_df = db.pull(query)
            
            logger.info(f"FACEIDMapping query completed: {len(faceid_df)} records found")
            
            if not faceid_df.empty:
                unique_ceids = faceid_df['CEID'].nunique()
                logger.info(f"Found FACEIDMapping data for {unique_ceids} CEIDs")
                
                # Log sample data
                sample_data = faceid_df.head(3)
                logger.debug("Sample FACEIDMapping data:")
                for _, row in sample_data.iterrows():
                    logger.debug(f"  CEID: {row['CEID']}, SGL4: {row['SGL4']}, SGL5: {row['SGL5']}, SGL6: {row['SGL6']}, SGL7: {row['SGL7']}")
            else:
                logger.warning("No FACEIDMapping data found for the provided CEIDs")
            
            return faceid_df
            
        except Exception as e:
            logger.error(f"Error executing FACEIDMapping query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def transform_faceid_to_contacts_current_shift(self, faceid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform FACEIDMapping data but only include CURRENT shift group leader
        Uses F_CALENDAR to determine current shift
        """
        if faceid_df.empty:
            logger.warning("Empty FACEIDMapping dataframe provided")
            return pd.DataFrame()
        
        # Get current shift information from F_CALENDAR
        shift_info = self.get_current_shift_info()
        
        if not shift_info['success']:
            logger.error("Failed to get current shift from F_CALENDAR, using fallback")
            current_shift = 'SGL4'  # Fallback
            shift_description = 'Fallback - Calendar lookup failed'
        else:
            current_shift = shift_info['sgl_id']  # SGL4, SGL5, SGL6, or SGL7
            shift_description = shift_info['shift_description']
        
        logger.info(f"Transforming contacts for CURRENT SHIFT from F_CALENDAR: {current_shift}")
        logger.info(f"Shift: {shift_description}")
        
        if shift_info['success']:
            logger.info(f"Calendar Shift: {shift_info['calendar_shift']}")
            logger.info(f"Active Period: {shift_info['start_date']} to {shift_info['end_date']}")
        
        all_contacts = []
        current_shift_found = 0
        current_shift_missing = 0
        
        for _, row in faceid_df.iterrows():
            ceid = row['CEID']
            
            # Add ONLY the current shift group leader (determined from F_CALENDAR)
            shift_name_col = current_shift  # e.g., 'SGL4'
            shift_wwid_col = f"{current_shift}WWID"  # e.g., 'SGL4WWID'
            
            if pd.notna(row[shift_name_col]) and pd.notna(row[shift_wwid_col]):
                all_contacts.append({
                    'CEID': ceid,
                    'CONTACT_ROLE': 'Shift_Group_Leader',
                    'CONTACT_NAME': str(row[shift_name_col]),
                    'WWID': str(row[shift_wwid_col]),
                    'SHIFT_ID': shift_name_col,
                    'EMAIL_ADDRESS': None,
                    'SOURCE': 'FACEIDMapping',
                    'IS_CURRENT_SHIFT': True,
                    'SHIFT_DESCRIPTION': shift_description,
                    'CALENDAR_SHIFT': shift_info.get('calendar_shift', 'Unknown'),
                    'SHIFT_START': shift_info.get('start_date', None),
                    'SHIFT_END': shift_info.get('end_date', None)
                })
                current_shift_found += 1
                logger.debug(f"Added current shift contact: {row[shift_name_col]} ({shift_name_col}) for CEID {ceid}")
            else:
                current_shift_missing += 1
                logger.warning(f"No {current_shift} contact found for CEID {ceid}")
                logger.debug(f"  {shift_name_col}: {row.get(shift_name_col, 'NULL')}")
                logger.debug(f"  {shift_wwid_col}: {row.get(shift_wwid_col, 'NULL')}")
            
            # Still add managers (they work across all shifts)
            # Department Manager (MgrGroup1)
            if pd.notna(row['MgrGroup1']) and pd.notna(row['MgrGroup1WWID']):
                all_contacts.append({
                    'CEID': ceid,
                    'CONTACT_ROLE': 'Department_Manager',
                    'CONTACT_NAME': str(row['MgrGroup1']),
                    'WWID': str(row['MgrGroup1WWID']),
                    'SHIFT_ID': None,
                    'EMAIL_ADDRESS': None,
                    'SOURCE': 'FACEIDMapping',
                    'IS_CURRENT_SHIFT': False,
                    'SHIFT_DESCRIPTION': 'All Shifts',
                    'CALENDAR_SHIFT': None,
                    'SHIFT_START': None,
                    'SHIFT_END': None
                })
            
            # Engineering Manager (MgrGroup3)
            if pd.notna(row['MgrGroup3']) and pd.notna(row['MgrGroup3WWID']):
                all_contacts.append({
                    'CEID': ceid,
                    'CONTACT_ROLE': 'Engineering_Manager',
                    'CONTACT_NAME': str(row['MgrGroup3']),
                    'WWID': str(row['MgrGroup3WWID']),
                    'SHIFT_ID': None,
                    'EMAIL_ADDRESS': None,
                    'SOURCE': 'FACEIDMapping',
                    'IS_CURRENT_SHIFT': False,
                    'SHIFT_DESCRIPTION': 'All Shifts',
                    'CALENDAR_SHIFT': None,
                    'SHIFT_START': None,
                    'SHIFT_END': None
                })
        
        if not all_contacts:
            logger.warning("No contacts extracted from FACEIDMapping data")
            return pd.DataFrame()
        
        contacts_df = pd.DataFrame(all_contacts)
        
        # Log transformation statistics
        logger.info(f"F_CALENDAR-based shift contact transformation completed:")
        logger.info(f"  Current shift ({current_shift}) contacts found: {current_shift_found}")
        logger.info(f"  Current shift ({current_shift}) contacts missing: {current_shift_missing}")
        logger.info(f"  Total contacts extracted: {len(contacts_df)}")
        
        if not contacts_df.empty:
            role_counts = contacts_df['CONTACT_ROLE'].value_counts().to_dict()
            logger.info(f"  Role distribution: {role_counts}")
            
            current_shift_contacts = contacts_df[contacts_df['IS_CURRENT_SHIFT'] == True]
            if not current_shift_contacts.empty:
                logger.info(f"  Current shift contacts: {len(current_shift_contacts)}")
                for _, contact in current_shift_contacts.iterrows():
                    logger.info(f"    - {contact['CONTACT_NAME']} ({contact['SHIFT_ID']}) for CEID {contact['CEID']}")
                    logger.info(f"      Calendar Shift: {contact['CALENDAR_SHIFT']}")
        
        return contacts_df

    def get_email_addresses_for_wwids(self, wwids: List[str]) -> pd.DataFrame:
        """
        Get email addresses and employee details for WWIDs from F_WORKER table in Xeus
        """
        if not wwids:
            logger.warning("No WWIDs provided for email lookup")
            return pd.DataFrame()
        
        clean_wwids = list(set([str(wwid) for wwid in wwids if wwid is not None and str(wwid) != 'nan']))
        
        if not clean_wwids:
            logger.warning("No valid WWIDs after cleaning")
            return pd.DataFrame()
        
        placeholders = ','.join([f"'{wwid}'" for wwid in clean_wwids])
        
        query = f"""
        SELECT 
            WWID,
            FIRST_NAME,
            LAST_NAME,
            FULL_NAME,
            CORPORATE_EMAIL,
            WORK_NUMBER,
            MOBILE_NUMBER,
            SHIFT,
            MANAGER_NAME,
            MANAGER_WWID,
            BUSINESS_TITLE,
            DEPARTMENT_NAME,
            STATUS,
            WORK_LOCATION
        FROM F_WORKER
        WHERE WWID IN ({placeholders})
          AND STATUS = 'ACTIVE'
        """
        
        try:
            logger.info(f"Fetching email addresses for {len(clean_wwids)} WWIDs from F_WORKER table in Xeus")
            logger.debug(f"WWIDs: {clean_wwids[:10]}...")
            
            # Connect to Xeus database (same as your main query)
            conn = PyUber.connect(datasource='F21_PROD_XEUS')
            cursor = conn.cursor()
            cursor.execute(query)
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            email_df = pd.DataFrame(results, columns=columns)
            
            logger.info(f"F_WORKER email query completed: {len(email_df)} records found")
            
            if not email_df.empty:
                # Log statistics
                active_employees = len(email_df)
                employees_with_email = email_df['CORPORATE_EMAIL'].notna().sum()
                
                logger.info(f"Found {active_employees} active employees in Xeus F_WORKER")
                logger.info(f"Employees with corporate email: {employees_with_email}")
                
                # Log sample data
                sample_data = email_df.head(3)
                logger.debug("Sample F_WORKER data from Xeus:")
                for _, row in sample_data.iterrows():
                    logger.debug(f"  WWID: {row['WWID']}, Name: {row['FULL_NAME']}, Email: {row['CORPORATE_EMAIL']}")
            else:
                logger.warning("No active employees found in Xeus F_WORKER for the provided WWIDs")
            
            return email_df
            
        except Exception as e:
            logger.error(f"Error executing F_WORKER email query from Xeus: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
            logger.info("F_WORKER Xeus connection closed")

    def get_enhanced_contacts_for_ceids(self, ceids: List[str], current_shift_only: bool = True) -> pd.DataFrame:
        """
        Enhanced contact lookup using:
        1. FACEIDMapping table from IEIndicators (CEIDs -> WWIDs + Names)
        2. F_WORKER table from Xeus (WWIDs -> Emails + Details)
        3. F_CALENDAR table from Xeus (Current Shift Detection)
        
        Args:
            ceids: List of CEIDs to lookup
            current_shift_only: If True, only gets current shift GL from F_CALENDAR. If False, gets all 4 shift GLs
        """
        if not ceids:
            logger.warning("No CEIDs provided for enhanced contacts lookup")
            return pd.DataFrame()
        
        shift_mode = "CURRENT SHIFT ONLY (F_CALENDAR)" if current_shift_only else "ALL SHIFTS"
        logger.info(f"Starting enhanced contacts lookup - {shift_mode}:")
        logger.info("  - FACEIDMapping from IEIndicators")
        logger.info("  - F_WORKER from Xeus")
        if current_shift_only:
            logger.info("  - F_CALENDAR from Xeus (shift detection)")
        
        # Step 1: Get FACEIDMapping data from IEIndicators
        faceid_df = self.get_faceid_mapping_contacts(ceids)
        
        if faceid_df.empty:
            logger.warning("No FACEIDMapping data found in IEIndicators")
            return pd.DataFrame()
        
        # Step 2: Transform to contacts format
        if current_shift_only:
            contacts_df = self.transform_faceid_to_contacts_current_shift(faceid_df)
            logger.info("Using CURRENT SHIFT ONLY contact transformation (F_CALENDAR)")
        else:
            contacts_df = self.transform_faceid_to_contacts(faceid_df)
            logger.info("Using ALL SHIFTS contact transformation")
        
        if contacts_df.empty:
            logger.warning("No contacts extracted from FACEIDMapping")
            return pd.DataFrame()
        
        # Step 3: Get email addresses from F_WORKER in Xeus
        unique_wwids = contacts_df['WWID'].unique().tolist()
        logger.info(f"Looking up emails for {len(unique_wwids)} unique WWIDs in Xeus F_WORKER")
        
        email_df = self.get_email_addresses_for_wwids(unique_wwids)
        
        if not email_df.empty:
            # Merge email and employee data
            logger.info("Merging Xeus F_WORKER data with IEIndicators contacts")
            
            # Select relevant columns from F_WORKER
            worker_cols = ['WWID', 'FIRST_NAME', 'LAST_NAME', 'FULL_NAME', 'CORPORATE_EMAIL', 
                          'WORK_NUMBER', 'MOBILE_NUMBER', 'SHIFT', 'BUSINESS_TITLE', 
                          'DEPARTMENT_NAME', 'STATUS', 'WORK_LOCATION']
            
            # Merge with contacts
            enhanced_contacts_df = contacts_df.merge(
                email_df[worker_cols],
                on='WWID',
                how='left',
                suffixes=('', '_WORKER')
            )
            
            # Update EMAIL_ADDRESS with CORPORATE_EMAIL
            enhanced_contacts_df['EMAIL_ADDRESS'] = enhanced_contacts_df['CORPORATE_EMAIL']
            
            # Log merge statistics
            total_contacts = len(contacts_df)
            contacts_with_emails = enhanced_contacts_df['EMAIL_ADDRESS'].notna().sum()
            
            logger.info(f"Cross-datasource email merge completed ({shift_mode}):")
            logger.info(f"  Total contacts from IEIndicators: {total_contacts}")
            logger.info(f"  Contacts with emails from Xeus: {contacts_with_emails}")
            logger.info(f"  Email coverage: {(contacts_with_emails/total_contacts*100):.1f}%")
            
            # Log current shift specific info
            if current_shift_only:
                current_shift_contacts = enhanced_contacts_df[enhanced_contacts_df['IS_CURRENT_SHIFT'] == True]
                if not current_shift_contacts.empty:
                    logger.info(f"  Current shift contacts with emails: {current_shift_contacts['EMAIL_ADDRESS'].notna().sum()}")
            
            # Log contacts without emails
            missing_emails = enhanced_contacts_df[enhanced_contacts_df['EMAIL_ADDRESS'].isna()]
            if not missing_emails.empty:
                missing_wwids = missing_emails['WWID'].unique()
                logger.warning(f"Contacts without emails in Xeus F_WORKER: {len(missing_wwids)} WWIDs")
                logger.debug(f"WWIDs without emails: {missing_wwids[:5]}...")
            
            return enhanced_contacts_df
        else:
            logger.warning("No email data found in Xeus F_WORKER, returning contacts without emails")
            return contacts_df

    def merge_contacts_with_data(self, df: pd.DataFrame, contacts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the main dataframe with enhanced contacts information including F_WORKER data
        """
        if df.empty:
            logger.warning("Empty main dataframe provided for contacts merge")
            return df
        
        if contacts_df.empty:
            logger.warning("Empty contacts dataframe provided for merge")
            # Add empty contact columns to maintain structure
            df['CONTACT_ROLE'] = None
            df['CONTACT_NAME'] = None
            df['WWID'] = None
            df['SHIFT_ID'] = None
            df['EMAIL_ADDRESS'] = None
            df['FIRST_NAME'] = None
            df['LAST_NAME'] = None
            df['FULL_NAME'] = None
            df['WORK_NUMBER'] = None
            df['MOBILE_NUMBER'] = None
            df['BUSINESS_TITLE'] = None
            df['DEPARTMENT_NAME'] = None
            df['WORK_LOCATION'] = None
            df['SOURCE'] = None
            df['IS_CURRENT_SHIFT'] = None
            df['SHIFT_DESCRIPTION'] = None
            df['CALENDAR_SHIFT'] = None
            df['SHIFT_START'] = None
            df['SHIFT_END'] = None
            return df
        
        if 'TARGET_CEID' not in df.columns:
            logger.error("TARGET_CEID column not found in main dataframe")
            return df
        
        logger.info("Merging enhanced contacts (FACEIDMapping + F_WORKER + F_CALENDAR) with main data...")
        
        # Perform left join to get all contacts for each TARGET_CEID
        merged_df = df.merge(
            contacts_df,
            left_on='TARGET_CEID',
            right_on='CEID',
            how='left'
        )
        
        # Drop the duplicate CEID column from contacts
        if 'CEID' in merged_df.columns:
            merged_df = merged_df.drop('CEID', axis=1)
        
        # Log merge statistics
        original_rows = len(df)
        merged_rows = len(merged_df)
        rows_with_contacts = merged_df['CONTACT_ROLE'].notna().sum()
        rows_with_emails = merged_df['EMAIL_ADDRESS'].notna().sum()
        
        logger.info(f"Enhanced contacts merge completed:")
        logger.info(f"  - Original rows: {original_rows}")
        logger.info(f"  - Merged rows: {merged_rows}")
        logger.info(f"  - Rows with contacts: {rows_with_contacts}")
        logger.info(f"  - Rows with emails: {rows_with_emails}")
        logger.info(f"  - Expansion factor: {merged_rows/original_rows:.2f}x")
        
        # Log contact role distribution
        if not merged_df.empty and 'CONTACT_ROLE' in merged_df.columns:
            role_counts = merged_df['CONTACT_ROLE'].value_counts().to_dict()
            logger.info(f"  - Contact role distribution: {role_counts}")
        
        return merged_df

    def compute_target_operation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute TARGET_OPERATION based on the business logic:
        - isMetro = Y, Operation != null and PROCESS_OPERATION != null: TARGET_OPERATION = PROCESS_OPERATION
        - isMetro = Y, Operation != null, PROCESS_OPERATION = null: TARGET_OPERATION = OPERATION
        - isMetro = N: TARGET_OPERATION = OPERATION
        """
        if df.empty:
            logger.warning("Empty dataframe provided for TARGET_OPERATION computation")
            return df
        
        logger.info("Computing TARGET_OPERATION column...")
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check for the isMetro column with different possible names
        metro_col = None
        possible_metro_cols = ['isMetro', 'ISMETRO', 'IsMetro', 'is_metro', 'IS_METRO']
        
        for col in possible_metro_cols:
            if col in df_copy.columns:
                metro_col = col
                logger.info(f"Found metro column: {metro_col}")
                break
        
        if metro_col is None:
            logger.error("Metro column not found in dataframe")
            logger.error(f"Available columns: {list(df_copy.columns)}")
            # Create isMetro column based on AREA if available
            if 'AREA' in df_copy.columns:
                logger.info("Creating isMetro column from AREA column")
                df_copy['isMetro'] = df_copy['AREA'].apply(
                    lambda x: 'Y' if str(x).upper() in ['METRO', 'ANALYTICAL'] else 'N'
                )
                metro_col = 'isMetro'
            else:
                logger.error("Cannot create isMetro column - AREA column not available")
                return df_copy
        
        # Standardize column names for easier processing
        if metro_col != 'isMetro':
            df_copy['isMetro'] = df_copy[metro_col]
            metro_col = 'isMetro'
        
        # Check for required columns
        required_cols = ['OPERATION', 'PROCESS_OPERATION']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns: {list(df_copy.columns)}")
            return df_copy
        
        # Initialize TARGET_OPERATION column
        df_copy['TARGET_OPERATION'] = None
        
        # Convert columns to string and handle NaN/None values
        df_copy['isMetro'] = df_copy['isMetro'].astype(str)
        df_copy['OPERATION'] = df_copy['OPERATION'].astype(str)
        df_copy['PROCESS_OPERATION'] = df_copy['PROCESS_OPERATION'].astype(str)
        
        # Replace 'nan', 'None', empty strings with actual None for easier checking
        df_copy['OPERATION'] = df_copy['OPERATION'].replace(['nan', 'None', '', 'NULL'], None)
        df_copy['PROCESS_OPERATION'] = df_copy['PROCESS_OPERATION'].replace(['nan', 'None', '', 'NULL'], None)
        
        # Apply the business logic
        conditions = [
            # Condition 1: isMetro = Y, Operation != null, PROCESS_OPERATION != null
            (df_copy['isMetro'] == 'Y') & 
            (df_copy['OPERATION'].notna()) & 
            (df_copy['PROCESS_OPERATION'].notna()),
            
            # Condition 2: isMetro = Y, Operation != null, PROCESS_OPERATION = null
            (df_copy['isMetro'] == 'Y') & 
            (df_copy['OPERATION'].notna()) & 
            (df_copy['PROCESS_OPERATION'].isna()),
            
            # Condition 3: isMetro = N (regardless of other conditions)
            (df_copy['isMetro'] == 'N')
        ]
        
        choices = [
            df_copy['PROCESS_OPERATION'],  # Use PROCESS_OPERATION
            df_copy['OPERATION'],          # Use OPERATION
            df_copy['OPERATION']           # Use OPERATION
        ]
        
        # Apply the conditions using numpy.select
        df_copy['TARGET_OPERATION'] = np.select(conditions, choices, default=df_copy['OPERATION'])
        
        # Log statistics about the computation
        metro_y_count = len(df_copy[df_copy['isMetro'] == 'Y'])
        metro_n_count = len(df_copy[df_copy['isMetro'] == 'N'])
        
        process_op_used = len(df_copy[
            (df_copy['isMetro'] == 'Y') & 
            (df_copy['OPERATION'].notna()) & 
            (df_copy['PROCESS_OPERATION'].notna())
        ])
        
        operation_used_metro = len(df_copy[
            (df_copy['isMetro'] == 'Y') & 
            (df_copy['OPERATION'].notna()) & 
            (df_copy['PROCESS_OPERATION'].isna())
        ])
        
        operation_used_non_metro = metro_n_count
        
        logger.info(f"TARGET_OPERATION computation completed:")
        logger.info(f"  - Metro lots (Y): {metro_y_count}")
        logger.info(f"  - Non-Metro lots (N): {metro_n_count}")
        logger.info(f"  - Used PROCESS_OPERATION: {process_op_used}")
        logger.info(f"  - Used OPERATION (Metro with null PROCESS_OPERATION): {operation_used_metro}")
        logger.info(f"  - Used OPERATION (Non-Metro): {operation_used_non_metro}")
        
        # Verify no null TARGET_OPERATION values
        null_targets = df_copy['TARGET_OPERATION'].isna().sum()
        if null_targets > 0:
            logger.warning(f"Found {null_targets} rows with null TARGET_OPERATION")
        else:
            logger.info("All rows have valid TARGET_OPERATION values")
        
        return df_copy

    def compute_target_ceid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute TARGET_CEID for each entry using TARGET_OPERATION
        """
        if df.empty:
            logger.warning("Empty dataframe provided for TARGET_CEID computation")
            return df
        
        if 'TARGET_OPERATION' not in df.columns:
            logger.error("TARGET_OPERATION column not found. Cannot compute TARGET_CEID")
            return df
        
        logger.info("Computing TARGET_CEID column...")
        
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Get unique target operations for CEID lookup
        unique_target_ops = df_copy['TARGET_OPERATION'].dropna().unique().tolist()
        logger.info(f"Found {len(unique_target_ops)} unique target operations for CEID lookup")
        
        # Get CEID mapping from database
        ceid_mapping_df = self.get_ceid_mapping(unique_target_ops)
        
        if ceid_mapping_df.empty:
            logger.warning("No CEID mappings found. Setting TARGET_CEID to None for all rows")
            df_copy['TARGET_CEID'] = None
            return df_copy
        
        # Create a dictionary for faster lookup
        # Handle cases where multiple CEIDs exist for the same operation
        ceid_dict = {}
        operation_ceid_counts = ceid_mapping_df.groupby('OPERATION')['CEID_LIST'].count()
        
        for operation, count in operation_ceid_counts.items():
            if count == 1:
                # Single CEID for this operation
                ceid = ceid_mapping_df[ceid_mapping_df['OPERATION'] == operation]['CEID_LIST'].iloc[0]
                ceid_dict[operation] = ceid
            else:
                # Multiple CEIDs for this operation - take the first one and log a warning
                ceids = ceid_mapping_df[ceid_mapping_df['OPERATION'] == operation]['CEID_LIST'].tolist()
                ceid_dict[operation] = ceids[0]  # Take the first one
                logger.warning(f"Multiple CEIDs found for operation '{operation}': {ceids}. Using: {ceids[0]}")
        
        logger.info(f"Created CEID mapping dictionary with {len(ceid_dict)} entries")
        
        # Map TARGET_OPERATION to TARGET_CEID
        df_copy['TARGET_CEID'] = df_copy['TARGET_OPERATION'].map(ceid_dict)
        
        # Log statistics
        total_rows = len(df_copy)
        mapped_rows = df_copy['TARGET_CEID'].notna().sum()
        unmapped_rows = total_rows - mapped_rows
        
        logger.info(f"TARGET_CEID computation completed:")
        logger.info(f"  - Total rows: {total_rows}")
        logger.info(f"  - Rows with TARGET_CEID: {mapped_rows}")
        logger.info(f"  - Rows without TARGET_CEID: {unmapped_rows}")
        
        if unmapped_rows > 0:
            unmapped_operations = df_copy[df_copy['TARGET_CEID'].isna()]['TARGET_OPERATION'].unique()
            logger.warning(f"Operations without CEID mapping: {unmapped_operations[:10]}...")  # Show first 10
        
        # Log some sample mappings
        sample_mapped = df_copy[df_copy['TARGET_CEID'].notna()][['TARGET_OPERATION', 'TARGET_CEID']].drop_duplicates().head(5)
        if not sample_mapped.empty:
            logger.info("Sample TARGET_OPERATION -> TARGET_CEID mappings:")
            for _, row in sample_mapped.iterrows():
                logger.info(f"  {row['TARGET_OPERATION']} -> {row['TARGET_CEID']}")
        
        return df_copy

    def build_email_subject(self, state: str) -> str:
        """Build email subject based on escalation state"""
        subject_mapping = {
            'State 2': 'URGENT: Lot Escalation - State 2 (Current Shift)',
            'State 3': 'CRITICAL: Lot Escalation - State 3 (Engineering)',
            'State 4': 'EMERGENCY: Lot Escalation - State 4 (Department Manager)'
        }
        return subject_mapping.get(state, f'Lot Escalation - {state}')

    def build_email_body(self, row) -> str:
        """Build HTML email body with table"""
        name = row['FULL_NAME'] if pd.notna(row['FULL_NAME']) else 'Team Member'
        shift_id = f" ({row['SHIFT_ID']})" if pd.notna(row['SHIFT_ID']) else ""
        
        state_messages = {
            'State 2': 'A lot requires immediate attention on your current shift:',
            'State 3': 'A lot has escalated to engineering level and requires your immediate attention:',
            'State 4': 'A lot has reached MAXIMUM escalation level and requires your immediate intervention:'
        }
        
        message = state_messages.get(row['STATE'], 'A lot requires your attention:')
        
        # Handle potential None/NaN values
        def safe_str(value):
            return str(value) if pd.notna(value) else 'N/A'
        
        body = f"""Hello {name}{shift_id},

{message}

<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
    <tr><td><b>LOT</b></td><td>{safe_str(row['LOT'])}</td></tr>
    <tr><td><b>OPERATION</b></td><td>{safe_str(row['OPERATION'])}</td></tr>
    <tr><td><b>PROCESS_OPERATION</b></td><td>{safe_str(row['PROCESS_OPERATION'])}</td></tr>
    <tr><td><b>AREA</b></td><td>{safe_str(row['AREA'])}</td></tr>
    <tr><td><b>MODULE</b></td><td>{safe_str(row['MODULE'])}</td></tr>
    <tr><td><b>HOURS AT OPERATION</b></td><td>{safe_str(row['HOURS_AT_OPERATION'])} hours</td></tr>
    <tr><td><b>STATE</b></td><td>{safe_str(row['STATE'])}</td></tr>
    <tr><td><b>TARGET_CEID</b></td><td>{safe_str(row['TARGET_CEID'])}</td></tr>
    <tr><td><b>ESCALATION_NODE</b></td><td>{safe_str(row['EMAIL_ADDRESS'])}</td></tr>
</table>

Please take immediate action.

Best regards,
NMPROD Automation Escalation System"""
        
        return body

    def send_escalation_emails(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Send escalation emails with ALL ACTIVE_EMAILS as CC"""
        if df.empty:
            logger.warning("No data provided for email sending")
            return {"success": False, "message": "No data to process"}
        
        logger.info("Starting escalation email sending process...")
        if self.test_mode:
            logger.info("TEST MODE: Emails will be sent to test recipients (rishitha.kondrolla@intel.com, uriel.mendiola@intel.com)")
        else:
            logger.info("PRODUCTION MODE: TO: Individual contacts, CC: ALL ACTIVE_EMAILS from RW_Department, BCC: rishitha.kondrolla@intel.com")
        
        # Get ALL ACTIVE_EMAILS from ALL departments (no mapping needed) - only for production mode
        all_active_emails = []
        if not self.test_mode:
            dept_df = self.get_rw_departments()
            all_active_emails = self.get_all_active_emails(dept_df)
            
            if all_active_emails:
                logger.info(f"Will CC {len(all_active_emails)} emails to all escalations: {all_active_emails}")
            else:
                logger.warning("No ACTIVE_EMAILS found in any department")
        
        stats = {
            'total_records': len(df),
            'emails_attempted': 0,
            'emails_sent': 0,
            'emails_failed': 0,
            'no_email_address': 0,
            'state_1_skipped': 0,
            'cc_emails_used': len(all_active_emails),
            'by_state': {'State 2': 0, 'State 3': 0, 'State 4': 0},
            'test_mode': self.test_mode
        }
        
        try:
            for _, row in df.iterrows():
                # Skip State 1 (non-escalation)
                if row['STATE'] == 'State 1':
                    stats['state_1_skipped'] += 1
                    continue
                
                # In production mode, skip if no email address for primary recipient
                if not self.test_mode and (pd.isna(row['EMAIL_ADDRESS']) or str(row['EMAIL_ADDRESS']).strip() == ''):
                    stats['no_email_address'] += 1
                    logger.warning(f"No email address for lot {row['LOT']}, operation {row['OPERATION']}, state {row['STATE']}")
                    continue
                
                # Get department/group name for display only
                group_name = row.get('GROUP_NAME', None)
                department = str(group_name) if pd.notna(group_name) else 'Unknown'
                
                # Build email content
                subject = self.build_email_subject(row['STATE'])
                body = self.build_email_body(row)
                
                # Send email
                stats['emails_attempted'] += 1
                
                success = self.email_service.send_escalation_email(
                    recipient_email=row['EMAIL_ADDRESS'] if not self.test_mode else "rishitha.kondrolla@intel.com",
                    subject=subject,
                    body=body,
                    recipient_name=row['FULL_NAME'] if pd.notna(row['FULL_NAME']) else 'Team Member',
                    state=row['STATE'],
                    cc_emails=all_active_emails if not self.test_mode else ["uriel.mendiola@intel.com"],
                    department=department
                )
                
                if success:
                    stats['emails_sent'] += 1
                    if row['STATE'] in stats['by_state']:
                        stats['by_state'][row['STATE']] += 1
                else:
                    stats['emails_failed'] += 1
            
            mode = "TEST MODE" if self.test_mode else "PRODUCTION MODE"
            logger.info(f"{mode} email sending completed:")
            logger.info(f"  Total records: {stats['total_records']}")
            logger.info(f"  State 1 skipped: {stats['state_1_skipped']}")
            logger.info(f"  Emails attempted: {stats['emails_attempted']}")
            logger.info(f"  Emails sent: {stats['emails_sent']}")
            logger.info(f"  Emails failed: {stats['emails_failed']}")
            logger.info(f"  No email address: {stats['no_email_address']}")
            logger.info(f"  CC emails used: {stats['cc_emails_used']}")
            logger.info(f"  By state: {stats['by_state']}")
            if not self.test_mode:
                logger.info(f"  BCC recipient (all emails): {self.email_service.bcc_recipient}")
            
            return {
                "success": True,
                "message": f"{mode} email sending completed",
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error in email sending process: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": str(e),
                "stats": stats
            }

    def insert_escalation_tracking(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Insert escalation records with enhanced contact information from F_CALENDAR + FACEIDMapping + F_WORKER
        Only inserts escalation records (State 2, 3, 4) - excludes State 1 (non-escalation)
        """
        if df.empty:
            logger.warning("Empty dataframe provided for escalation tracking")
            return {"success": False, "message": "No data to process"}
        
        if self.test_mode:
            logger.info("TEST MODE: Skipping escalation tracking database insertion")
            return {"success": True, "message": "Test mode - no database insertion", "stats": {}}
        
        logger.info("Starting escalation tracking insertion with F_CALENDAR shift detection...")
        logger.info("Note: State 1 (non-escalation) records will be skipped")
        
        # Define escalation mapping based on state - ONLY for escalation states
        escalation_mapping = {
            'State 2': {'role': 'Shift_Group_Leader', 'contact_role_filter': 'Shift_Group_Leader'},
            'State 3': {'role': 'Engineering_Manager', 'contact_role_filter': 'Engineering_Manager'},
            'State 4': {'role': 'Department_Manager', 'contact_role_filter': 'Department_Manager'}
        }
        
        # Required columns check - UPDATED to include GROUP_NAME
        required_columns = ['LOT', 'OPERATION', 'STATE', 'TARGET_CEID', 'PROCESS_OPERATION', 'GROUP_NAME']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns for escalation tracking: {missing_columns}")
            return {"success": False, "message": f"Missing columns: {missing_columns}"}
        
        # Check for isMetro column
        metro_col = None
        possible_metro_cols = ['isMetro', 'ISMETRO', 'IsMetro', 'is_metro', 'IS_METRO']
        for col in possible_metro_cols:
            if col in df.columns:
                metro_col = col
                break
        
        if metro_col is None:
            logger.warning("Metro column not found, will set IsMetro to NULL")
        
        escalation_records = []
        stats = {
            'total_processed': 0,
            'state_1_skipped': 0,
            'state_2_records': 0,
            'state_3_records': 0,
            'state_4_records': 0,
            'records_with_contacts': 0,
            'records_without_contacts': 0,
            'records_with_emails': 0,
            'current_shift_escalations': 0,
            'duplicates_skipped': 0,
            'new_records_inserted': 0
        }
        
        try:
            with F11xDB() as db:
                # Get current user for CREATED_BY
                current_user = getpass.getuser()
                current_time = datetime.now()
                
                # Process each row in the dataframe
                for idx, row in df.iterrows():
                    stats['total_processed'] += 1
                    
                    lot = str(row['LOT']) if pd.notna(row['LOT']) else None
                    operation = str(row['OPERATION']) if pd.notna(row['OPERATION']) else None
                    state = str(row['STATE']) if pd.notna(row['STATE']) else None
                    target_ceid = str(row['TARGET_CEID']) if pd.notna(row['TARGET_CEID']) else None
                    process_operation = str(row['PROCESS_OPERATION']) if pd.notna(row['PROCESS_OPERATION']) else None
                    group_name = str(row['GROUP_NAME']) if pd.notna(row['GROUP_NAME']) else None  # ADDED
                    
                    # Skip if essential fields are missing
                    if not all([lot, operation, state]):
                        logger.warning(f"Skipping row {idx}: missing essential fields (LOT, OPERATION, or STATE)")
                        continue
                    
                    # Skip State 1 (non-escalation)
                    if state == 'State 1':
                        stats['state_1_skipped'] += 1
                        logger.debug(f"Skipping State 1 (non-escalation) for lot {lot}, operation {operation}")
                        continue
                    
                    # Get IsMetro value
                    is_metro = None
                    if metro_col and pd.notna(row[metro_col]):
                        is_metro = 1 if str(row[metro_col]).upper() == 'Y' else 0
                    
                    # Get escalation info for this state
                    if state not in escalation_mapping:
                        logger.warning(f"Unknown escalation state '{state}' for lot {lot}, operation {operation}")
                        continue
                    
                    escalation_info = escalation_mapping[state]
                    role = escalation_info['role']
                    contact_role_filter = escalation_info['contact_role_filter']
                    
                    # Count state occurrences
                    if state == 'State 2':
                        stats['state_2_records'] += 1
                    elif state == 'State 3':
                        stats['state_3_records'] += 1
                    elif state == 'State 4':
                        stats['state_4_records'] += 1
                    
                    # Get contacts based on role
                    if 'CONTACT_ROLE' in df.columns and 'WWID' in df.columns:
                        # Filter contacts for this specific row and role
                        row_contacts = df[
                            (df['LOT'] == lot) & 
                            (df['OPERATION'] == operation) & 
                            (df['STATE'] == state) &
                            (df['CONTACT_ROLE'] == contact_role_filter)
                        ]
                        
                        if not row_contacts.empty:
                            # Create escalation record for each contact
                            for _, contact_row in row_contacts.iterrows():
                                escalated_to = str(contact_row['WWID']) if pd.notna(contact_row['WWID']) else 'Unknown'
                                wwid = escalated_to
                                
                                # Use CORPORATE_EMAIL if available, otherwise EMAIL_ADDRESS
                                email_address = None
                                if pd.notna(contact_row.get('CORPORATE_EMAIL')):
                                    email_address = str(contact_row['CORPORATE_EMAIL'])
                                elif pd.notna(contact_row.get('EMAIL_ADDRESS')):
                                    email_address = str(contact_row['EMAIL_ADDRESS'])
                                
                                # Get full name if available
                                full_name = None
                                if pd.notna(contact_row.get('FULL_NAME')):
                                    full_name = str(contact_row['FULL_NAME'])
                                elif pd.notna(contact_row.get('CONTACT_NAME')):
                                    full_name = str(contact_row['CONTACT_NAME'])
                                
                                # Get shift information
                                shift_id = str(contact_row.get('SHIFT_ID', '')) if pd.notna(contact_row.get('SHIFT_ID')) else None
                                calendar_shift = str(contact_row.get('CALENDAR_SHIFT', '')) if pd.notna(contact_row.get('CALENDAR_SHIFT')) else None
                                is_current_shift = contact_row.get('IS_CURRENT_SHIFT', False)
                                
                                escalation_record = {
                                    'LOT': lot,
                                    'OPERATION': operation,
                                    'STATE': state,
                                    'ESCALATED_TO': full_name or escalated_to,
                                    'WWID': wwid,
                                    'PROCESS_OPERATION': process_operation,
                                    'TARGET_CEID': target_ceid,
                                    'ESCALATED_DATE': current_time,
                                    'ESCALATION_NODE': email_address or 'No_Email',
                                    'CREATED_BY': current_user,
                                    'ROLE': role,
                                    'IsMetro': is_metro,
                                    'RW_Department': group_name  # ADDED
                                }
                                
                                escalation_records.append(escalation_record)
                                stats['records_with_contacts'] += 1
                                
                                if email_address:
                                    stats['records_with_emails'] += 1
                                
                                if is_current_shift:
                                    stats['current_shift_escalations'] += 1
                                    logger.debug(f"Current shift escalation: {full_name or escalated_to} ({shift_id}) for lot {lot}")
                            
                        else:
                            # No contacts found for this role, create a placeholder record
                            escalation_record = {
                                'LOT': lot,
                                'OPERATION': operation,
                                'STATE': state,
                                'ESCALATED_TO': f'No_{role}_Found',
                                'WWID': None,
                                'PROCESS_OPERATION': process_operation,
                                'TARGET_CEID': target_ceid,
                                'ESCALATED_DATE': current_time,
                                'ESCALATION_NODE': 'System',
                                'CREATED_BY': current_user,
                                'ROLE': role,
                                'IsMetro': is_metro,
                                'RW_Department': group_name  # ADDED
                            }
                            
                            escalation_records.append(escalation_record)
                            stats['records_without_contacts'] += 1
                    else:
                        # No contact columns available, create placeholder
                        escalation_record = {
                            'LOT': lot,
                            'OPERATION': operation,
                            'STATE': state,
                            'ESCALATED_TO': f'No_Contacts_Available',
                            'WWID': None,
                            'PROCESS_OPERATION': process_operation,
                            'TARGET_CEID': target_ceid,
                            'ESCALATED_DATE': current_time,
                            'ESCALATION_NODE': 'System',
                            'CREATED_BY': current_user,
                            'ROLE': role,
                            'IsMetro': is_metro,
                            'RW_Department': group_name  # ADDED
                        }
                        
                        escalation_records.append(escalation_record)
                        stats['records_without_contacts'] += 1
                
                logger.info(f"Prepared {len(escalation_records)} escalation records for insertion")
                logger.info(f"Records with emails: {stats['records_with_emails']}")
                logger.info(f"Current shift escalations: {stats['current_shift_escalations']}")
                
                if not escalation_records:
                    logger.warning("No escalation records to insert")
                    return {"success": True, "message": "No escalation records to insert", "stats": stats}
                
                # Insert records (only if they don't exist) - UPDATED SQL TO INCLUDE RW_Department
                insert_sql = """
                INSERT INTO NPI_ESCALATION_TRACKING 
                (LOT, OPERATION, STATE, ESCALATED_TO, WWID, PROCESS_OPERATION, TARGET_CEID, 
                 ESCALATED_DATE, ESCALATION_NODE, CREATED_BY, ROLE, IsMetro, RW_Department)
                SELECT :LOT, :OPERATION, :STATE, :ESCALATED_TO, :WWID, :PROCESS_OPERATION, :TARGET_CEID,
                       :ESCALATED_DATE, :ESCALATION_NODE, :CREATED_BY, :ROLE, :IsMetro, :RW_Department
                WHERE NOT EXISTS (
                    SELECT 1 FROM NPI_ESCALATION_TRACKING 
                    WHERE LOT = :LOT 
                      AND OPERATION = :OPERATION 
                      AND STATE = :STATE 
                      AND ESCALATED_TO = :ESCALATED_TO
                )
                """
                
                # Execute insertions
                with db.engine.connect() as conn:
                    with conn.begin():  # Use transaction
                        for record in escalation_records:
                            try:
                                result = conn.execute(text(insert_sql), record)
                                if result.rowcount > 0:
                                    stats['new_records_inserted'] += 1
                                else:
                                    stats['duplicates_skipped'] += 1
                            except Exception as e:
                                logger.error(f"Error inserting record {record}: {e}")
                                continue
                
                logger.info("F_CALENDAR-based escalation tracking insertion completed successfully")
                logger.info(f"Statistics: {stats}")
                
                return {
                    "success": True,
                    "message": "F_CALENDAR-based escalation tracking completed",
                    "stats": stats
                }
                
        except Exception as e:
            logger.error(f"Error in F_CALENDAR-based escalation tracking insertion: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": str(e),
                "stats": stats
            }

    def process_results(self, df: pd.DataFrame, current_shift_only: bool = True) -> Dict[str, Any]:
        """Process and analyze the results with F_CALENDAR shift detection"""
        if df.empty:
            logger.warning("No data to process")
            return {"success": False, "message": "No data retrieved"}
        
        # First, compute TARGET_OPERATION
        df = self.compute_target_operation(df)
        
        # Then, compute TARGET_CEID
        df = self.compute_target_ceid(df)
        
        # Get enhanced contacts from multiple datasources
        unique_ceids = df['TARGET_CEID'].dropna().unique().tolist()
        if unique_ceids:
            shift_mode = "current shift only (F_CALENDAR)" if current_shift_only else "all shifts"
            logger.info(f"Fetching enhanced contacts for {len(unique_ceids)} unique CEIDs ({shift_mode})")
            contacts_df = self.get_enhanced_contacts_for_ceids(unique_ceids, current_shift_only=current_shift_only)
            
            # Merge contacts with main data
            df = self.merge_contacts_with_data(df, contacts_df)
        else:
            logger.warning("No TARGET_CEIDs found for contacts lookup")
            # Add empty contact columns
            df['CONTACT_ROLE'] = None
            df['CONTACT_NAME'] = None
            df['WWID'] = None
            df['SHIFT_ID'] = None
            df['EMAIL_ADDRESS'] = None
            df['FIRST_NAME'] = None
            df['LAST_NAME'] = None
            df['FULL_NAME'] = None
            df['WORK_NUMBER'] = None
            df['MOBILE_NUMBER'] = None
            df['BUSINESS_TITLE'] = None
            df['DEPARTMENT_NAME'] = None
            df['WORK_LOCATION'] = None
            df['SOURCE'] = None
            df['IS_CURRENT_SHIFT'] = None
            df['SHIFT_DESCRIPTION'] = None
            df['CALENDAR_SHIFT'] = None
            df['SHIFT_START'] = None
            df['SHIFT_END'] = None
        
        analysis = {
            "success": True,
            "total_rows": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "dataframe": df,
            "current_shift_only": current_shift_only
        }
        
        # Add current shift information to analysis
        if current_shift_only:
            try:
                shift_info = self.get_current_shift_info()
                analysis["current_shift_info"] = shift_info
            except Exception as e:
                logger.error(f"Error getting current shift info: {e}")
        
        # Group analysis
        if 'GROUP_NAME' in df.columns:
            group_summary = df.groupby('GROUP_NAME').size().reset_index(name='Count')
            analysis["group_summary"] = group_summary.to_dict('records')
            logger.info(f"Data distributed across {len(group_summary)} groups")
        
        # State analysis
        if 'STATE' in df.columns:
            state_counts = df['STATE'].value_counts().to_dict()
            analysis["state_distribution"] = state_counts
            logger.info(f"State distribution: {state_counts}")
        
        # Metro analysis
        metro_col = 'isMetro' if 'isMetro' in df.columns else None
        if not metro_col:
            for col in ['ISMETRO', 'IsMetro', 'is_metro', 'IS_METRO']:
                if col in df.columns:
                    metro_col = col
                    break
        
        if metro_col:
            metro_counts = df[metro_col].value_counts().to_dict()
            analysis["metro_distribution"] = metro_counts
            logger.info(f"Metro distribution: {metro_counts}")
        
        # TARGET_OPERATION analysis
        if 'TARGET_OPERATION' in df.columns:
            target_op_counts = df['TARGET_OPERATION'].value_counts().head(10).to_dict()
            analysis["target_operation_top10"] = target_op_counts
            logger.info(f"Top 10 TARGET_OPERATIONS: {list(target_op_counts.keys())}")
            
            # Analysis of which source was used for TARGET_OPERATION
            if metro_col:
                metro_y_df = df[df[metro_col] == 'Y']
                if not metro_y_df.empty:
                    used_process_op = len(metro_y_df[
                        (metro_y_df['OPERATION'].notna()) & 
                        (metro_y_df['PROCESS_OPERATION'].notna())
                    ])
                    used_operation = len(metro_y_df[
                        (metro_y_df['OPERATION'].notna()) & 
                        (metro_y_df['PROCESS_OPERATION'].isna())
                    ])
                    
                    analysis["target_operation_source"] = {
                        "metro_used_process_operation": used_process_op,
                        "metro_used_operation": used_operation,
                        "non_metro_used_operation": len(df[df[metro_col] == 'N'])
                    }
        
        # TARGET_CEID analysis
        if 'TARGET_CEID' in df.columns:
            ceid_mapped_count = df['TARGET_CEID'].notna().sum()
            ceid_unmapped_count = df['TARGET_CEID'].isna().sum()
            
            analysis["target_ceid_summary"] = {
                "mapped_count": int(ceid_mapped_count),
                "unmapped_count": int(ceid_unmapped_count),
                "mapping_rate": float(ceid_mapped_count / len(df) * 100) if len(df) > 0 else 0
            }
            
            if ceid_mapped_count > 0:
                target_ceid_counts = df['TARGET_CEID'].value_counts().head(10).to_dict()
                analysis["target_ceid_top10"] = target_ceid_counts
                logger.info(f"Top 10 TARGET_CEIDs: {list(target_ceid_counts.keys())}")
        
        # Enhanced Contacts analysis
        if 'CONTACT_ROLE' in df.columns:
            contacts_count = df['CONTACT_ROLE'].notna().sum()
            
            analysis["contacts_summary"] = {
                "rows_with_contacts": int(contacts_count),
                "total_contact_records": int(contacts_count),
                "coverage_rate": float(contacts_count / len(df) * 100) if len(df) > 0 else 0
            }
            
            if contacts_count > 0:
                contact_role_counts = df['CONTACT_ROLE'].value_counts().to_dict()
                analysis["contact_role_distribution"] = contact_role_counts
                logger.info(f"Contact role distribution: {contact_role_counts}")
                
                # Email coverage analysis
                if 'EMAIL_ADDRESS' in df.columns:
                    emails_count = df['EMAIL_ADDRESS'].notna().sum()
                    analysis["email_coverage"] = {
                        "contacts_with_emails": int(emails_count),
                        "email_coverage_rate": float(emails_count / contacts_count * 100) if contacts_count > 0 else 0
                    }
                
                # Current shift analysis
                if current_shift_only and 'IS_CURRENT_SHIFT' in df.columns:
                    current_shift_contacts = df[df['IS_CURRENT_SHIFT'] == True]
                    analysis["current_shift_contacts"] = {
                        "count": len(current_shift_contacts),
                        "with_emails": current_shift_contacts['EMAIL_ADDRESS'].notna().sum() if not current_shift_contacts.empty else 0
                    }
                
                # Shift analysis
                if 'SHIFT_ID' in df.columns:
                    shift_counts = df[df['CONTACT_ROLE'] == 'Shift_Group_Leader']['SHIFT_ID'].value_counts().to_dict()
                    analysis["shift_distribution"] = shift_counts
                    logger.info(f"Shift leader distribution: {shift_counts}")
                
                # Count unique contacts per CEID
                if 'TARGET_CEID' in df.columns:
                    contacts_per_ceid = df[df['CONTACT_ROLE'].notna()].groupby('TARGET_CEID')['WWID'].nunique().describe().to_dict()
                    analysis["contacts_per_ceid_stats"] = contacts_per_ceid
        
        return analysis

    def run_full_process(self, current_shift_only: bool = True, send_emails: bool = False) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the complete data processing pipeline - simplified without department matching"""
        
        shift_mode = "CURRENT SHIFT ONLY (F_CALENDAR)" if current_shift_only else "ALL SHIFTS"
        
        logger.info("="*60)
        logger.info(f"STARTING F_CALENDAR ENHANCED XEUS PIPELINE - {shift_mode}")
        logger.info("Multi-datasource: F_CALENDAR + IEIndicators + Xeus + F11x")  # Added F11x back for ACTIVE_EMAILS
        if self.test_mode:
            logger.info("RUNNING IN TEST MODE - No database writes")
            if send_emails:
                logger.info("TEST MODE EMAILS ENABLED - Sending to test recipients only")
        if send_emails and not self.test_mode:
            logger.info("EMAIL SENDING ENABLED with CC/BCC support")
        
        if current_shift_only:
            try:
                shift_info = self.get_current_shift_info()
                if shift_info['success']:
                    logger.info(f"Current Shift: {shift_info['calendar_shift']} -> {shift_info['sgl_id']}")
                    logger.info(f"Description: {shift_info['shift_description']}")
                    logger.info(f"Active Period: {shift_info['start_date']} to {shift_info['end_date']}")
                else:
                    logger.warning(f"Shift detection failed: {shift_info['message']}")
            except Exception as e:
                logger.error(f"Error getting shift info: {e}")
        logger.info("="*60)
        
        try:
            # Step 1: Execute query directly (simple filter for F11X NPI department)
            dept_filter = "UPPER(F_RW_DEPT.DEPT_NAME) LIKE UPPER('%F11X NPI%')"
            
            result_df = self.execute_xeus_query(dept_filter)
            
            # Step 2: Process results (includes TARGET_OPERATION, TARGET_CEID, and enhanced contacts)
            analysis = self.process_results(result_df, current_shift_only=current_shift_only)
            
            # Get the processed dataframe with all computed columns
            if analysis["success"] and "dataframe" in analysis:
                result_df = analysis["dataframe"]
            
            # Step 3: Insert escalation tracking records (only for States 2, 3, 4)
            if not result_df.empty:
                logger.info(f"Starting F_CALENDAR-based escalation tracking insertion ({shift_mode})...")
                escalation_result = self.insert_escalation_tracking(result_df)
                analysis["escalation_tracking"] = escalation_result
                
                if escalation_result["success"]:
                    logger.info("F_CALENDAR-based escalation tracking completed successfully")
                else:
                    logger.error(f"F_CALENDAR-based escalation tracking failed: {escalation_result['message']}")
            else:
                logger.warning("No data available for escalation tracking")
                analysis["escalation_tracking"] = {"success": False, "message": "No data to process"}
            
            # Step 4: Send emails if requested (with ALL ACTIVE_EMAILS as CC)
            if send_emails and not result_df.empty:
                if self.test_mode:
                    logger.info("Starting TEST MODE escalation email sending to test recipients...")
                else:
                    logger.info("Starting escalation email sending with ALL ACTIVE_EMAILS as CC...")
                email_result = self.send_escalation_emails(result_df)
                analysis["email_sending"] = email_result
                
                if email_result["success"]:
                    if self.test_mode:
                        logger.info("TEST MODE email sending completed successfully")
                    else:
                        logger.info("Email sending with CC/BCC completed successfully")
                else:
                    logger.error(f"Email sending failed: {email_result['message']}")
            elif send_emails:
                logger.warning("No data available for email sending")
                analysis["email_sending"] = {"success": False, "message": "No data to process"}
            else:
                logger.info("Email sending disabled")
                analysis["email_sending"] = {"success": True, "message": "Email sending disabled"}
            
            logger.info("="*60)
            logger.info(f"F_CALENDAR ENHANCED PIPELINE COMPLETED SUCCESSFULLY - {shift_mode}")
            logger.info("="*60)
            
            return result_df, analysis
            
        except Exception as e:
            logger.error(f"F_CALENDAR enhanced pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(), {"success": False, "message": str(e)}

def display_results(df: pd.DataFrame, analysis: Dict[str, Any]):
    """Display results in a formatted way including F_CALENDAR shift information"""
    print("\n" + "="*60)
    print("F_CALENDAR ENHANCED RESULTS SUMMARY")
    print("="*60)
    
    # Display current shift information from F_CALENDAR
    if "current_shift_info" in analysis:
        shift_info = analysis["current_shift_info"]
        if shift_info['success']:
            print(f"CURRENT SHIFT INFORMATION (from F_CALENDAR):")
            print(f"  Calendar Shift: {shift_info['calendar_shift']}")
            print(f"  Mapped to SGL: {shift_info['sgl_id']}")
            print(f"  Description: {shift_info['shift_description']}")
            print(f"  Current Time: {shift_info['current_system_time']}")
            print(f"  Shift Period: {shift_info['start_date']} to {shift_info['end_date']}")
            print(f"  Status: {shift_info['shift_status']}")
            if shift_info['total_active_shifts'] > 1:
                print(f"  Multiple active shifts found: {shift_info['total_active_shifts']}")
        else:
            print(f"F_CALENDAR SHIFT LOOKUP FAILED: {shift_info['message']}")
    
    if analysis["success"]:
        shift_mode = "CURRENT SHIFT ONLY (F_CALENDAR)" if analysis.get("current_shift_only", True) else "ALL SHIFTS"
        print(f"SUCCESS: {analysis['total_rows']} rows retrieved ({shift_mode})")
        print(f"Columns: {len(analysis['columns'])}")
        
        if not df.empty:
            print("\nFIRST 5 ROWS:")
            print(df.head().to_string())
            
            if "group_summary" in analysis:
                print("\nGROUP SUMMARY:")
                for group in analysis["group_summary"]:
                    print(f"  {group['GROUP_NAME']}: {group['Count']} lots")
            
            if "state_distribution" in analysis:
                print("\nSTATE DISTRIBUTION:")
                for state, count in analysis["state_distribution"].items():
                    print(f"  {state}: {count}")
            
            if "metro_distribution" in analysis:
                print("\nMETRO DISTRIBUTION:")
                for metro, count in analysis["metro_distribution"].items():
                    metro_label = "Metro" if metro == 'Y' else "Non-Metro"
                    print(f"  {metro_label}: {count}")
            
            # Enhanced Contacts analysis display
            if "contacts_summary" in analysis:
                contacts_info = analysis["contacts_summary"]
                print(f"\nENHANCED CONTACTS SUMMARY ({shift_mode}):")
                print(f"  Rows with contacts: {contacts_info['rows_with_contacts']}")
                print(f"  Total contact records: {contacts_info['total_contact_records']}")
                print(f"  Coverage rate: {contacts_info['coverage_rate']:.1f}%")
            
            # Current shift specific information
            if "current_shift_contacts" in analysis:
                current_shift_info = analysis["current_shift_contacts"]
                print(f"\nCURRENT SHIFT CONTACTS (from F_CALENDAR):")
                print(f"  Current shift contacts: {current_shift_info['count']}")
                print(f"  Current shift with emails: {current_shift_info['with_emails']}")
            
            if "email_coverage" in analysis:
                email_info = analysis["email_coverage"]
                print("\nEMAIL COVERAGE (from Xeus F_WORKER):")
                print(f"  Contacts with emails: {email_info['contacts_with_emails']}")
                print(f"  Email coverage rate: {email_info['email_coverage_rate']:.1f}%")
            
            if "contact_role_distribution" in analysis:
                print("\nCONTACT ROLE DISTRIBUTION:")
                for role, count in analysis["contact_role_distribution"].items():
                    print(f"  {role}: {count}")
            
            if "shift_distribution" in analysis:
                print("\nSHIFT LEADER DISTRIBUTION:")
                for shift, count in analysis["shift_distribution"].items():
                    print(f"  {shift}: {count}")
        
        # Add F_CALENDAR-based escalation tracking results
        if "escalation_tracking" in analysis:
            escalation_info = analysis["escalation_tracking"]
            print(f"\nF_CALENDAR-BASED ESCALATION TRACKING RESULTS ({shift_mode}):")
            
            if escalation_info["success"]:
                stats = escalation_info.get("stats", {})
                print(f"  Total records processed: {stats.get('total_processed', 0)}")
                print(f"  State 1 records skipped (non-escalation): {stats.get('state_1_skipped', 0)}")
                print(f"  New escalation records inserted: {stats.get('new_records_inserted', 0)}")
                print(f"  Duplicates skipped: {stats.get('duplicates_skipped', 0)}")
                print(f"  Records with contacts: {stats.get('records_with_contacts', 0)}")
                print(f"  Records with emails: {stats.get('records_with_emails', 0)}")
                print(f"  Current shift escalations: {stats.get('current_shift_escalations', 0)}")
                print(f"  Records without contacts: {stats.get('records_without_contacts', 0)}")
                
                print("\nESCALATION STATE BREAKDOWN:")
                print(f"  State 2 (Current Shift GL from F_CALENDAR): {stats.get('state_2_records', 0)}")
                print(f"  State 3 (Engineering Manager): {stats.get('state_3_records', 0)}")
                print(f"  State 4 (Department Manager): {stats.get('state_4_records', 0)}")
            else:
                print(f"  FAILED: {escalation_info.get('message', 'Unknown error')}")
        
        # Add email sending results with CC/BCC info
        if "email_sending" in analysis:
            email_info = analysis["email_sending"]
            test_mode = email_info.get("stats", {}).get("test_mode", False)
            mode = "TEST MODE" if test_mode else "PRODUCTION MODE"
            print(f"\n{mode} EMAIL SENDING RESULTS:")
            
            if email_info["success"]:
                stats = email_info.get("stats", {})
                if stats:  # Only show if stats exist (not disabled)
                    print(f"  Total records: {stats.get('total_records', 0)}")
                    print(f"  State 1 skipped: {stats.get('state_1_skipped', 0)}")
                    print(f"  Emails attempted: {stats.get('emails_attempted', 0)}")
                    print(f"  Emails sent successfully: {stats.get('emails_sent', 0)}")
                    print(f"  Emails failed: {stats.get('emails_failed', 0)}")
                    print(f"  No email address: {stats.get('no_email_address', 0)}")
                    print(f"  CC emails used: {stats.get('cc_emails_used', 0)}")
                    print(f"  By state: {stats.get('by_state', {})}")
                    if test_mode:
                        print(f"  Test recipients: rishitha.kondrolla@intel.com (TO), uriel.mendiola@intel.com (CC)")
                    else:
                        print(f"  BCC recipient (all emails): rishitha.kondrolla@intel.com")
                else:
                    print(f"  {email_info.get('message', 'Email sending completed')}")
            else:
                print(f"  FAILED: {email_info.get('message', 'Unknown error')}")
                
    else:
        print(f"FAILED: {analysis.get('message', 'Unknown error')}")
    
    print("="*60)

def main(current_shift_only: bool = True, test_mode: bool = False, send_emails: bool = False, test_mode_with_emails: bool = False):
    """Main execution function for F_CALENDAR-based escalation system with enhanced SMTP support"""
    
    # Override send_emails if test_mode_with_emails is True
    if test_mode_with_emails:
        test_mode = True
        send_emails = True
        logger.info("TEST MODE WITH EMAILS ENABLED - Emails will be sent to test recipients only")
    
    processor = XeusDataProcessor(test_mode=test_mode)
    result_df, analysis = processor.run_full_process(current_shift_only=current_shift_only, send_emails=send_emails)
    
    display_results(result_df, analysis)
    
    return result_df, analysis

if __name__ == "__main__":
    # Example usage:
    
    # New test mode - sends emails to test recipients only, no database writes
    final_df, final_analysis = main(current_shift_only=True, test_mode_with_emails=True)
    
    # Optional: Save results to file
    if not final_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"f_calendar_shift_results_{timestamp}.csv"
        final_df.to_csv(filename, index=False)
        logger.info(f"F_CALENDAR shift results saved to {filename}")
