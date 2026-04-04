# SmartOps n8n Workflows

These workflows connect SmartOps to business automation tools.

## Workflows

### 1. manual_query.json
Manually trigger a question to SmartOps and see the answer.
Use case: Testing and ad-hoc queries.

### 2. scheduled_report.json  
Runs every morning at 8am, queries SmartOps, emails the summary.
Use case: Automated daily business reports.

### 3. external_webhook.json
Exposes a webhook URL that any external tool can POST to.
Use case: Slack bots, form submissions, third-party integrations.

## Setup
- SmartOps backend must be running on port 8000
- Inside Docker, use http://smartops:8000 as the base URL
- Outside Docker, use http://localhost:8000