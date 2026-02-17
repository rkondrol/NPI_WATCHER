# NPI_WATCHER
The NPI Watcher is an automated escalation system designed to monitor and manage NPI lots that become stagnated at metrology tools. Many lots get stuck at metrology operations due to various issues including SPC problems or other operational bottlenecks.
Solution
This system implements a time-based escalation workflow that automatically routes stagnated lots to appropriate personnel based on wait duration:
Escalation Tiers
< 2 hours: No escalation (State 1)
2-8 hours: Escalated to Metrology CEID Shift Group Leader (State 2)
8-12 hours: Escalated to Engineering Manager (State 3)
12+ hours: Escalated to Department Manager (State 4)
Key Features
Real-time Monitoring: Tracks lot duration at each metrology operation
Automated Escalation: Time-based routing to appropriate management levels
Metro/Analytical Focus: Specifically targets metrology and analytical areas
SPC Integration: Handles both SPC and non-SPC related issues
CEID Tracking: Monitors both operation and process CEID lists
Comment System: Maintains audit trail with comments and timestamps
Technical Implementation
The system queries production databases to:
Calculate lot wait times at operations
Identify metrology/analytical area lots
Determine appropriate CEID targets
Track escalation states
Maintain historical comments and edits
