## 1️⃣ Executive Overview (Board / Leadership Level)
### What We Built
The Document Portal application is deployed on AWS using:
- **Amazon ECS (Fargate)** for container orchestration
- **Amazon ECR** for container image storage
- **GitHub Actions** for CI/CD automation
- **Direct Public IP exposure** (no load balancer)

The system automatically deploys new container versions when code is pushed to main and tests pass.

### High-Level Architecture
```
Developer → GitHub → CI (tests)
                         ↓
                     CD pipeline
                         ↓
                     ECR image
                         ↓
                 ECS Service (1 task)
                         ↓
                 Public IP : 8080
                         ↓
                      Internet
```

### Key Characteristics
```
| Component       | Value                                        |
| --------------- | -------------------------------------------- |
| Cluster         | `document-portal-cluster`                    |
| Service         | `documentportal-rebuild-td-service-zo7ca2lm` |
| Desired Tasks   | 1                                            |
| Deployment Type | Rolling Update                               |
| Public Access   | Direct Public IP                             |
| Container Port  | 8080                                         |

```

### Executive-Level Explanation
- The system runs one container instance at a time
- When new code is deployed, ECS safely replaces the old container with a new one
- The service is reachable via a public IP address on port 8080
- The IP changes during each deployment

This architecture is cost-efficient and ideal for development or internal systems

## 2️⃣ System Architecture (Engineering-Level View)
### Compute Layer
#### ECS Cluster
Logical container management environment:
```css
document-portal-cluster
```

### ECS Service
Controls task lifecycle:
```css
documentportal-rebuild-td-service-zo7ca2lm
```

Service configuration:
- Desired count: 1
- Deployment strategy: Rolling update
- Min healthy: 100%
- Max running: 200%

This means during deployment:
- ECS temporarily runs 2 tasks
- New task becomes healthy
- Old task is terminated
- Ends with 1 running task

### Task Definition

Example:
```css
documentportal-rebuild-td:9
```

Task settings:
- Launch type: FARGATE
- Network mode: awsvpc
- CPU: 1 vCPU
- Memory: 8GB
- Container port: 8080
- Secrets injected from AWS Secrets Manager
- Logs sent to CloudWatch

Networking Layer
Each Fargate task gets:
- Its own ENI (Elastic Network Interface)
- A private IP
- A public IP (because Assign Public IP = ENABLED)

Example from deployment:
```
| Resource       | Value                    |
| -------------- | ------------------------ |
| Public IP      | 44.193.0.171             |
| Private IP     | 172.31.8.74              |
| Subnet         | subnet-0b0dbbf504123d36f |
| Security Group | sg-08e7d3d1135f661f3     |
```

Traffic Flow:
```
Internet
   ↓
Public IP:8080
   ↓
Security Group (allows TCP 8080)
   ↓
Task ENI
   ↓
Container (port 8080)
```

There is **no load balancer** in this architecture.

## 3️⃣ Deep Technical Layer (Advanced Understanding)
### Why the Public IP Changes
Each deployment:
1. New task starts
2. New ENI is created
3. New public IP assigned
4. Old task terminated
5. Old IP disappears

Because:
- Fargate tasks are ephemeral
- ENIs are tied to task lifecycle
- There is no persistent load balancer

### Deployment Lifecycle (Step-by-Step)
### CI Phase
`ci.yaml:`
- Runs pytest
- Ensures code quality

### CD Phase
`aws.yaml:`
1. Authenticate to AWS
2. Login to ECR
3. Build Docker image
4. Push image tagged with commit SHA
5. Render ECS task definition with new image
6. Deploy updated task definition to ECS service

### Rolling Update Mechanics
With Desired = 1
Min = 100%
Max = 200%

Deployment looks like:
```python
Before deploy:
[ Task v8 ]

During deploy:
[ Task v8 ] + [ Task v9 ]

After healthy:
[ Task v9 ]
```

Safe zero-downtime replacement.

### Security Model
- Public ingress allowed on port 8080
- Security group controls inbound traffic
- Secrets injected securely via Secrets Manager
- IAM Execution Role allows:
    - Pulling images from ECR
    - Writing logs to CloudWatch
    - Reading secrets

### Architecture Classification
#### Current State
- ✔ Single container
- ✔ Rolling updates
- ✔ Public IP exposure
- ✔ Secrets management
- ✔ Automated CI/CD

#### Not Yet Implemented
- Load Balancer (ALB)
- HTTPS termination
- Route53 DNS
- Auto scaling > 1 task
- WAF
- Blue/Green deployment

## 4️⃣ Operational Notes
### How to Access the Application
`http://<public_ip>:8080`

Note:
> Public IP changes after each deployment.

To find it:
> ECS → Service → Tasks → Networking tab → Public IP

## 5️⃣ Upgrade Path to Production Architecture
### Future improvement:
```java

Internet
   ↓
Application Load Balancer (HTTPS)
   ↓
ECS Service (multiple tasks)
   ↓
Fargate tasks
```

Benefits:
- Stable DNS
- HTTPS via ACM
- Auto scaling
- Health checks
- Zero IP changes
- Better resilience



