import os
import json
import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = None

# ==========================================
# Pydantic Models for Structured Output
# ==========================================

class AppLog(BaseModel):
    timestamp: str = Field(description="ISO 8601 timestamp")
    service: str = Field(description="Name of the microservice (e.g., auth-service, payment-gateway)")
    level: str = Field(description="Log level: INFO, WARN, ERROR, DEBUG")
    environment: str = Field(description="Environment: production, staging, dev, or a CI run ID like ci-pr-1234")
    trace_id: str = Field(description="A simulated distributed trace ID")
    message: str = Field(description="The actual log message, looking like a real application log")

class K8sEvent(BaseModel):
    timestamp: str = Field(description="ISO 8601 timestamp")
    kind: str = Field(description="Kubernetes Resource Kind (e.g., Pod, Deployment)")
    name: str = Field(description="Name of the resource")
    environment: str = Field(description="Environment: production, staging, dev, or a CI run ID like ci-pr-1234")
    reason: str = Field(description="Event reason (e.g., OOMKilled, BackOff, ScalingReplicaSet)")
    message: str = Field(description="Event message")

class PagerDutyAlert(BaseModel):
    id: str = Field(description="Unique alert ID")
    created_at: str = Field(description="ISO 8601 timestamp")
    service_affected: str = Field(description="The main service affected")
    environment: str = Field(description="Environment: production, staging, dev, or a CI run ID like ci-pr-1234")
    title: str = Field(description="Alert title")
    description: str = Field(description="Detailed alert description")
    urgency: str = Field(description="high or low")

class IncidentScenarioOutput(BaseModel):
    scenario_name: str
    is_error: bool
    app_logs: List[AppLog]
    k8s_events: List[K8sEvent]
    alert: Optional[PagerDutyAlert] = Field(description="Present only if is_error is true")

# ==========================================
# Scenarios
# ==========================================

SCENARIOS = [
    {
        "name": "Healthy Traffic - Payment Flow",
        "is_error": False,
        "description": "Normal, healthy traffic flowing through an API Gateway -> Auth Service -> Payment Service over a 5 minute window. Standard INFO logs, database queries succeeding in normal time."
    },
    {
        "name": "Incident: OOMKilled Image Processing",
        "is_error": True,
        "description": "An image-processing-worker service slowly consumes more memory over 10 minutes. Warning logs show high garbage collection times, followed by the process suddenly stopping. A Kubernetes OOMKilled event fires, resulting in a high urgency PagerDuty alert."
    },
    {
        "name": "Incident: Database Connection Pool Exhaustion",
        "is_error": True,
        "description": "An inventory-service experiences a sudden spike in traffic. Logs show connection acquisition timeouts to Postgres. Eventually throwing 500s. K8s events might show Liveness probe failures. Triggers a High Error Rate alert."
    },
    {
        "name": "Incident: Upstream API 504 Gateway Timeout",
        "is_error": True,
        "description": "The checkout-service relies on a third-party payment provider. The provider starts taking 10+ seconds to respond, causing context deadline exceeded errors in checkout-service. Results in a Latency alert."
    },
    {
        "name": "Incident: Bad Configuration Deploy",
        "is_error": True,
        "description": "A new version of auth-service is deployed. K8s events show the new pods starting. Immediately, the logs show 'Failed to load config: invalid JSON' and CrashLoopBackOff. Triggers a Deployment Failed alert."
    },
    {
        "name": "Incident: Cache Stampede (Redis down)",
        "is_error": True,
        "description": "Redis goes down briefly. The user-profile-service loses its cache and all requests hit the database simultaneously. DB latency spikes massively. Logs show Redis connection refused, then DB query timeouts."
    }
]

# ==========================================
# Generation Logic
# ==========================================

async def generate_scenario(scenario_def) -> IncidentScenarioOutput:
    print(f"Generating data for scenario: {scenario_def['name']}...", flush=True)
    
    base_time = datetime.now() - timedelta(hours=1)
    time_context = f"The events should happen roughly around {base_time.isoformat()}Z."
    
    prompt = f"""
    You are an expert Site Reliability Engineer generating highly realistic mock observability data for a system.
    
    Scenario Description: {scenario_def['description']}
    Is Error Scenario: {scenario_def['is_error']}
    Time Context: {time_context}
    
    Requirements:
    1. Generate between 15 to 30 Application logs (`app_logs`) that tell a clear chronological story.
    2. Generate 0 to 5 Kubernetes events (`k8s_events`) that make sense for this scenario.
    3. If this is an error scenario (`is_error` is true), generate exactly 1 PagerDuty alert (`alert`) that would be triggered by this incident. If it's a healthy scenario, `alert` MUST be null.
    4. Ensure timestamps are strictly chronological and make logical sense.
    5. Use realistic trace IDs (e.g., hex strings) to tie logs across different services together if applicable.
    6. Log messages should look like real JSON structured logs or standard application output (e.g. "msg='Connecting to db' host='pg-main' ms=12")
    7. Pick ONE environment (e.g., 'production', 'staging', 'dev', 'ci-pr-123') and use it consistently for all logs, events, and the alert in this scenario.
    """

    response = await client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IncidentScenarioOutput,
            temperature=0.7,
        ),
    )
    
    return IncidentScenarioOutput.model_validate_json(response.text)

def generate_background_noise_logs(start_time: datetime, end_time: datetime, num_logs: int) -> List[dict]:
    """Generates a large volume of realistic but uninteresting background logs."""
    services = ["api-gateway", "auth-service", "payment-service", "inventory-service", "user-profile-service", "checkout-service", "image-processing-worker"]
    levels = ["INFO", "DEBUG"]
    
    # Common mundane messages
    messages = [
        "Health check OK",
        "GET /health status=200",
        "Flushing metrics",
        "Pinging database",
        "msg='Connected to Redis' ms=2",
        "Refreshing configuration",
        "msg='Request processed' status=200 ms={ms}",
        "msg='Fetching user profile' cache=hit",
        "Background job 'cleanup' starting",
        "Background job 'cleanup' finished",
        "Garbage collection triggered",
    ]
    
    noise_logs = []
    time_diff_seconds = (end_time - start_time).total_seconds()
    
    for i in range(num_logs):
        # Pick a random time between start and end
        random_seconds = random.uniform(0, time_diff_seconds)
        log_time = start_time + timedelta(seconds=random_seconds)
        
        service = random.choice(services)
        level = "INFO" if random.random() > 0.2 else "DEBUG"
        msg = random.choice(messages)
        
        if "{ms}" in msg:
            msg = msg.format(ms=random.randint(5, 45))
            
        # Pick an environment (mostly production)
        r = random.random()
        if r < 0.7:
            env = "production"
        elif r < 0.9:
            env = "staging"
        elif r < 0.95:
            env = "dev"
        else:
            env = f"ci-pr-{random.randint(1000, 9999)}"
            
        noise_logs.append({
            "timestamp": log_time.isoformat() + "Z",
            "service": service,
            "level": level,
            "environment": env,
            "trace_id": uuid.uuid4().hex[:16],
            "message": msg,
            "_scenario_id": "background-noise"
        })
        
    return noise_logs

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment.")
        return
        
    global client
    client = genai.Client(api_key=api_key)

    os.makedirs("data", exist_ok=True)
    
    all_logs = []
    all_k8s_events = []
    all_alerts = []
    
    # Process scenarios sequentially to avoid rate limits on free tier, 
    # but could be parallelized if using a paid tier.
    for i, scenario in enumerate(SCENARIOS):
        try:
            output = await generate_scenario(scenario)
            
            # Append a scenario ID to trace them back easily during testing
            scenario_id = f"scenario-{i}"
            
            for log in output.app_logs:
                log_dict = log.model_dump()
                log_dict['_scenario_id'] = scenario_id
                all_logs.append(log_dict)
                
            for event in output.k8s_events:
                event_dict = event.model_dump()
                event_dict['_scenario_id'] = scenario_id
                all_k8s_events.append(event_dict)
                
            if output.alert:
                alert_dict = output.alert.model_dump()
                alert_dict['_scenario_id'] = scenario_id
                all_alerts.append(alert_dict)
                
        except Exception as e:
            print(f"Failed to generate scenario '{scenario['name']}': {e}")
            
    # Generate massive background noise to bury the needle in the haystack
    print(f"\nGenerating 50,000 background noise logs to simulate a realistic haystack...", flush=True)
    base_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now()
    noise_logs = generate_background_noise_logs(base_time, end_time, 50000)
    all_logs.extend(noise_logs)

    # Sort chronologically
    all_logs.sort(key=lambda x: x['timestamp'])
    all_k8s_events.sort(key=lambda x: x['timestamp'])

    # Save to files
    with open("data/logs.json", "w") as f:
        json.dump(all_logs, f, indent=2)
        
    with open("data/k8s_events.json", "w") as f:
        json.dump(all_k8s_events, f, indent=2)
        
    with open("data/alerts.json", "w") as f:
        json.dump(all_alerts, f, indent=2)
        
    print(f"\nSuccessfully generated mock data in 'data/' directory.")
    print(f"Generated {len(all_logs)} application logs.")
    print(f"Generated {len(all_k8s_events)} Kubernetes events.")
    print(f"Generated {len(all_alerts)} PagerDuty alerts.")

if __name__ == "__main__":
    asyncio.run(main())
