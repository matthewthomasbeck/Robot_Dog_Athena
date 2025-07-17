const express = require('express');
const axios = require('axios');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
const net = require('net');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: true,
    credentials: true
  }
});

// CORS middleware FIRST
app.use(cors({
  origin: true, // Reflects the request's origin
  credentials: true,
}));

// Explicitly handle OPTIONS preflight requests for CORS
app.options('/auth/token', cors({
  origin: true,
  credentials: true,
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Proxy endpoint for Cognito token exchange
app.post('/auth/token', async (req, res) => {
  try {
    const { code, redirectUri } = req.body;
    const params = new URLSearchParams();
    params.append('grant_type', 'authorization_code');
    params.append('client_id', process.env.COGNITO_CLIENT_ID);
    params.append('client_secret', process.env.COGNITO_CLIENT_SECRET);
    params.append('code', code);
    params.append('redirect_uri', redirectUri);

    const response = await axios.post(
        `https://${process.env.COGNITO_DOMAIN}/oauth2/token`,
        params,
        { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
    );
    res.json(response.data);
  } catch (err) {
    res.status(400).json({ error: err.response?.data || err.message });
  }
});

// Robot bridge functionality
const connectedClients = new Map();
const robotConnections = new Map();
let robotSocket = null;
let robotConnected = false;
let robotBuffer = Buffer.alloc(0);
let activeClientId = null; // Track which client is currently controlling the robot
let activeSessionId = null; // Track which session is currently controlling the robot

// Create robot server on port 3000
const robotServer = net.createServer((socket) => {
  logger.info('Robot connected from:', socket.remoteAddress);
  robotSocket = socket;
  robotConnected = true;

  // Notify the active client that robot is available (if there is one)
  if (activeClientId) {
    io.to(activeClientId).emit('robot-available');
  }
});

robotServer.listen(3000, '0.0.0.0', () => {
  logger.info('Robot server listening on port 3000');
});

// Add this function near the top or before the io.on('connection') block
function sanitizeKeys(keys) {
  // keys: array of strings
  const opposites = [
    ['w', 's'],
    ['a', 'd'],
    ['arrowup', 'arrowdown'],
    ['arrowleft', 'arrowright']
  ];
  let keySet = new Set(keys);
  opposites.forEach(([k1, k2]) => {
    if (keySet.has(k1) && keySet.has(k2)) {
      keySet.delete(k1);
      keySet.delete(k2);
    }
  });
  return Array.from(keySet);
}

// WebSocket signaling for WebRTC
io.on('connection', (socket) => {
  logger.info('Client connected:', socket.id);

  // Handle authentication
  socket.on('auth', async (data) => {
    try {
      // Verify the JWT token
      const token = data.token;
      if (!token) {
        socket.emit('auth-failed', { message: 'No token provided' });
        return;
      }

      // Extract session ID from JWT token (using sub claim as session identifier)
      const payload = JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString());
      const sessionId = payload.sub; // Use the user's unique ID as session identifier

      // Check if this session is already controlling the robot
      if (activeSessionId && activeSessionId === sessionId) {
        // This session is already active, allow this connection
        connectedClients.set(socket.id, { authenticated: true, token, videoReady: false, sessionId });
        socket.emit('auth-success');
        logger.info('Client authenticated (existing session):', socket.id);

        // Update active client ID to this new socket
        activeClientId = socket.id;

        // If robot is already connected, notify client immediately
        if (robotConnected) {
          socket.emit('robot-available');
        }
        return;
      }

      // Check if robot is already in use by another session
      if (activeSessionId && activeSessionId !== sessionId) {
        socket.emit('robot-in-use', { message: 'Robot is currently in use by another user.' });
        return;
      }

      // For now, just check if token exists (you can add more validation later)
      connectedClients.set(socket.id, { authenticated: true, token, videoReady: false, sessionId });
      socket.emit('auth-success');
      logger.info('Client authenticated:', socket.id);

      // Set this client as the active client and session
      activeClientId = socket.id;
      activeSessionId = sessionId;

      // If robot is already connected, notify client immediately
      if (robotConnected) {
        socket.emit('robot-available');
      }

    } catch (error) {
      logger.error('Authentication error:', error);
      socket.emit('auth-failed', { message: 'Authentication failed' });
    }
  });

  // Handle WebRTC signaling
  socket.on('offer', (data) => {
    logger.info('Received offer from client');

    const client = connectedClients.get(socket.id);
    if (client && client.authenticated && activeClientId === socket.id) {
      client.videoReady = true;

      // Send back an answer (simplified for now)
      setTimeout(() => {
        socket.emit('answer', {
          answer: {
            type: 'answer',
            sdp: 'v=0\r\no=- 0 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\na=msid-semantic: WMS\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 0.0.0.0\r\na=mid:0\r\na=sendonly\r\na=rtpmap:96 H264/90000\r\n'
          }
        });
      }, 1000);
    }
  });

  socket.on('answer', (data) => {
    logger.info('Received answer from client');
  });

  socket.on('ice-candidate', (data) => {
    logger.info('Received ICE candidate from client');
    // Echo back for testing
    socket.emit('ice-candidate', data);
  });

  // Handle robot commands
  socket.on('robot-command', (data) => {
    logger.info('Robot command received:', data.command);

    // Only allow commands from the active client
    if (activeClientId !== socket.id) {
      socket.emit('command-ack', {
        command: data.command,
        status: 'unauthorized',
        message: 'You are not the active controller',
        timestamp: Date.now()
      });
      return;
    }

    // Sanitize keys if the command is a key combination
    let commandToSend = data.command;
    if (typeof commandToSend === 'string' && commandToSend.includes('+')) {
      let keys = commandToSend.split('+');
      keys = sanitizeKeys(keys);
      commandToSend = keys.join('+');
    }

    // Forward command to robot if connected
    if (robotSocket && robotConnected) {
      try {
        // Send command with length prefix (matching robot's listen_for_commands format)
        const commandBytes = Buffer.from(commandToSend, 'utf8');
        const lengthBuffer = Buffer.alloc(4);
        lengthBuffer.writeUInt32BE(commandBytes.length, 0);
        robotSocket.write(lengthBuffer);
        robotSocket.write(commandBytes);
        socket.emit('command-ack', {
          command: commandToSend,
          status: 'sent',
          timestamp: Date.now()
        });
        logger.info(`Command sent to robot: ${commandToSend}`);
      } catch (error) {
        logger.error('Error sending command to robot:', error);
        socket.emit('command-ack', {
          command: commandToSend,
          status: 'error',
          error: error.message,
          timestamp: Date.now()
        });
      }
    } else {
      socket.emit('command-ack', {
        command: commandToSend,
        status: 'robot_disconnected',
        timestamp: Date.now()
      });
    }
  });

  // Handle client leaving (manual disconnect)
  socket.on('leave-robot', () => {
    logger.info('Client leaving robot control:', socket.id);
    if (activeClientId === socket.id) {
      activeClientId = null;
      activeSessionId = null;
      // Notify all authenticated clients that robot is now available
      connectedClients.forEach((client, clientId) => {
        if (client.authenticated) {
          io.to(clientId).emit('robot-available');
        }
      });
    }
  });

  socket.on('disconnect', () => {
    logger.info('Client disconnected:', socket.id);

    const client = connectedClients.get(socket.id);

    // If this was the active client, clear the active client and notify others
    if (activeClientId === socket.id) {
      activeClientId = null;
      activeSessionId = null;
      logger.info('Active client disconnected, robot is now available');

      // Notify all remaining authenticated clients that robot is available
      connectedClients.forEach((client, clientId) => {
        if (client.authenticated && clientId !== socket.id) {
          io.to(clientId).emit('robot-available');
        }
      });
    }

    connectedClients.delete(socket.id);
  });
});

// Add at the top, after other variable declarations
const MAX_FRAME_SIZE = 2 * 1024 * 1024; // 2MB max frame size for sanity check

// Log buffer size every 10 seconds
setInterval(() => {
  if (robotBuffer.length > 0) {
    logger.info('Current robotBuffer size:', robotBuffer.length);
  }
}, 10000); // every 10 seconds

// Log memory usage every minute
setInterval(() => {
  const mem = process.memoryUsage();
  logger.info('Memory usage:', mem);
}, 60000); // every minute

// Handle robot socket events
robotServer.on('connection', (socket) => {
  socket.on('data', (data) => {
    logger.info(`Received ${data.length} bytes from robot`);

    // Append new data to buffer
    robotBuffer = Buffer.concat([robotBuffer, data]);

    // Process complete frames
    while (robotBuffer.length >= 4) {
      // Read frame length (4 bytes)
      const frameLength = robotBuffer.readUInt32BE(0);

      // Sanity check for frame length
      if (frameLength <= 0 || frameLength > MAX_FRAME_SIZE) {
        logger.error('Invalid frame length:', frameLength, 'Clearing buffer.');
        robotBuffer = Buffer.alloc(0);
        break;
      }

      // Check if we have a complete frame
      if (robotBuffer.length >= 4 + frameLength) {
        // Extract frame data
        const frameData = robotBuffer.slice(4, 4 + frameLength);

        // Forward video data only to the active client
        if (activeClientId) {
          const activeClient = connectedClients.get(activeClientId);
          if (activeClient && activeClient.authenticated && activeClient.videoReady) {
            try {
              // The robot sends raw frame data (likely JPEG)
              // Convert to base64 for frontend consumption
              const frameBase64 = frameData.toString('base64');

              // Only send if we have valid data
              if (frameBase64.length > 0) {
                io.to(activeClientId).emit('video-frame', {
                  frame: frameBase64,
                  timestamp: Date.now()
                });
                logger.info(`Sent video frame to active client ${activeClientId}, size: ${frameLength} bytes`);
              } else {
                logger.warn('Frame base64 is empty, skipping emit.');
              }
            } catch (error) {
              logger.error('Error processing video frame:', error);
            }
          }
        }

        // Remove processed frame from buffer
        robotBuffer = robotBuffer.slice(4 + frameLength);
      } else {
        // Incomplete frame, wait for more data
        break;
      }
    }
  });

  socket.on('close', () => {
    logger.info('Robot disconnected');
    robotConnected = false;
    robotSocket = null;
    robotBuffer = Buffer.alloc(0); // Clear buffer on disconnect

    // Notify the active client that robot is unavailable
    if (activeClientId) {
      io.to(activeClientId).emit('robot-unavailable');
    }
  });

  socket.on('error', (err) => {
    logger.error('Robot socket error:', err);
    robotConnected = false;
    robotSocket = null;
    robotBuffer = Buffer.alloc(0); // Clear buffer on error

    // Notify the active client that robot is unavailable
    if (activeClientId) {
      io.to(activeClientId).emit('robot-unavailable');
    }
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    clients: connectedClients.size,
    activeClient: activeClientId,
    activeSession: activeSessionId,
    robotConnected: robotConnected,
    robotPort: 3000,
    webPort: process.env.PORT || 3001
  });
});

const PORT = process.env.PORT || 3001;
server.listen(PORT, '0.0.0.0', () => logger.info(`Backend running on port ${PORT}`));