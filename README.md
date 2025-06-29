#Project Athena!

##CREATOR(s):

Hi, my name is Matthew Beck, and I am responsible for writing specifically the code for this robot dog. Other extremely
important contributors to this project as a whole are Omar Ferrer, a fellow software engineer who helped me create the
environment (wouldn't have been able to do it without him) that allowed me to do this in the first place, and Aaed Musa,
the mechanical engineer responsible fot the Ares robotic dog platform, which has given me the luxury to focus on
software and not engineering.

##PURPOSE:

Software Engineering has an uneasy future ahead of it, and I wanted to make sure I had sharp skills in a way that
ensured said stills wouldn't immediately be replaced by A.I. (my thought process being that though it's not impossible
for a roboticist to be replaced by A.I., it's probably going to be a lot harder to replace me as compared to if I chose
to make apps and whatnot; sorry if I am accidentally throwing shade to some people, it's just what I think may happen).

This robot dog is meant to provide me a stepping stone of which I can move onto other projects such as:

* more advanced ML models
* controlling the robot via the internet
* more advanced robots

Hopefully these things pan out, we'll see.

##HOW YOU CAN HELP:

If you like it, STAR IT!!! I am a uni student right now, but I've got to start preparing for either a master's program
or to get hired, and so any help would mean the world.

# Robot Dog Control System

This system allows remote control of a robot dog via a web interface. The system consists of three main components:

1. **Robot Dog** - The physical robot that captures video and executes commands
2. **EC2 Backend** - A bridge server that handles communication between the robot and frontend
3. **Web Frontend** - A web interface for viewing video and sending commands

## System Architecture

```
[Robot Dog] ←→ [EC2 Backend (Port 3000)] ←→ [Web Frontend] ←→ [EC2 Backend (Port 3001)]
     ↑                    ↑                        ↑                    ↑
  Video + Commands    Bridge Logic            User Interface      Authentication
```

## Port Configuration

- **Port 3000**: Robot communication (TCP socket)
- **Port 3001**: Web frontend communication (HTTP/WebSocket)

## Setup Instructions

### 1. EC2 Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd <your website>-backend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables in `.env`:
   ```
   COGNITO_CLIENT_ID=your_cognito_client_id
   COGNITO_CLIENT_SECRET=your_cognito_client_secret
   COGNITO_DOMAIN=your_cognito_domain
   PORT=3001
   ```

4. Start the backend server:
   ```bash
   npm start
   ```

The backend will:
- Listen for robot connections on port 3000
- Handle web frontend connections on port 3001
- Bridge video data from robot to frontend
- Forward commands from frontend to robot
- Maintain authentication system

### 2. Robot Dog Setup

1. Ensure the robot is configured to connect to your EC2 instance:
   - Update `INTERNET_CONFIG` in `Robot_Dog/utilities/config.py`
   - Set `EC2_PUBLIC_IP` to your EC2 instance's public IP
   - Set `EC2_PORT` to 3000

2. Run the robot control logic:
   ```bash
   cd Robot_Dog
   python3 control_logic.py
   ```

The robot will:
- Connect to the EC2 backend on port 3000
- Stream video frames continuously
- Listen for commands from the frontend
- Execute movement commands using the existing control system

### 3. Web Frontend Setup

1. Deploy the frontend files to your web server
2. Ensure the frontend can connect to your EC2 backend

## Usage

### Accessing the Control Interface

1. Navigate to the controller page on your website
2. Log in with credentials that have 'owner' or 'privileged' group access
3. Click "Connect" to establish connection with the robot
4. Use the keyboard controls to operate the robot

### Robot Controls

- **W/↑** - Move Forward
- **S/↓** - Move Backward
- **A/←** - Turn Left
- **D/→** - Turn Right
- **Space** - Neutral Position
- **Q** - Exit

### Video Streaming

The system streams video from the robot's camera to the web interface in real-time. Video frames are:
1. Captured by the robot's camera
2. Sent to the EC2 backend via TCP socket
3. Converted to base64 and forwarded to the frontend via WebSocket
4. Displayed on an HTML5 canvas element

## Troubleshooting

### Connection Issues

1. **Robot not connecting**: Check that port 3000 is open on your EC2 instance
2. **Frontend not connecting**: Verify port 3001 is accessible and SSL certificates are valid
3. **Authentication failures**: Ensure Cognito configuration is correct

### Video Issues

1. **No video stream**: Check robot camera setup and frame data format
2. **Poor video quality**: Adjust camera resolution in robot config
3. **Video lag**: Consider reducing frame rate or video quality

### Command Issues

1. **Commands not working**: Verify robot is in 'web' mode
2. **Delayed response**: Check network latency between components
3. **Robot not responding**: Ensure robot control logic is running properly

## Security

- All web communication uses HTTPS
- Authentication is handled via AWS Cognito
- Only users in 'owner' or 'privileged' groups can access the controller
- Robot commands are validated before execution

## Development

### Adding New Commands

1. Add command handling in `Robot_Dog/control_logic.py` in the `_execute_keyboard_commands` function
2. Update the frontend keyboard event listeners in `<your website>-frontend/js/controller.js`
3. Add command documentation to the control instructions

### Modifying Video Format

1. Update the robot's camera utilities to output the desired format
2. Modify the backend frame processing in `<your website>-backend/server.js`
3. Update the frontend video display logic if needed

## Support

For issues or questions, check the logs in:
- Robot: `~/Robot_Dog/robot_dog.log`
- Backend: Console output or PM2 logs
- Frontend: Browser developer console