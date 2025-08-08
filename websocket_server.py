import asyncio
import websockets
import json
import threading
import time
from typing import Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserPresenceWebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.current_user_status = {
            'user_present': False,
            'user_count': 0,
            'last_detection_time': None,
            'distance': None,
            'gender': None,
            'age': None
        }
        self.server = None
        self.loop = None
        self.interval_task = None
        self.interval_broadcasting = False
        self.broadcast_interval = 30  # seconds
        self.last_immediate_broadcast = 0
        self.immediate_broadcast_throttle = 2.0  # minimum 2 seconds between immediate broadcasts
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send current status to new client
        await self.send_to_client(websocket, self.current_user_status)
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")
        
    async def send_to_client(self, websocket, data):
        """Send data to a specific client"""
        try:
            message = json.dumps(data)
            logger.info(f"Sending to client: {data}")
            await websocket.send(message)
            logger.info("Message sent successfully")
        except websockets.exceptions.ConnectionClosed:
            logger.error("Client connection closed while sending")
            self.clients.discard(websocket)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            
    async def broadcast_to_all(self, data):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
            
        message = json.dumps(data)
        disconnected = set()
        
        for client in self.clients.copy():
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
        
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                try:
                    print(f"message2..... {message}")
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await self.send_to_client(websocket, {
                        'error': 'Invalid JSON format'
                    })
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
            
    async def handle_client_message(self, websocket, data):
        """Handle messages from clients"""
        logger.info(f"Received message from client: {data}")
        
        # Handle both {"type": "ping"} and {"data": "ping"} formats
        message_type = data.get('type') or data.get('data')
        logger.info(f"Extracted message type: {message_type}")
        
        if message_type == 'ping':
            logger.info("Processing ping command")
            await self.send_to_client(websocket, {
                'type': 'pong',
                'timestamp': time.time()
            })
            logger.info("Pong response sent")
        elif message_type == 'pong':
            logger.info("Received pong from client")
            # Just acknowledge the pong, no response needed
        elif message_type == 'get_status':
            await self.send_to_client(websocket, {
                'type': 'status_update',
                **self.current_user_status
            })
        elif message_type == 'get_config':
            await self.send_to_client(websocket, {
                'type': 'config_response',
                'broadcast_interval': self.broadcast_interval,
                'interval_broadcasting': self.interval_broadcasting,
                'throttle_interval': self.immediate_broadcast_throttle
            })
        elif message_type == 'set_interval':
            # Allow client to change broadcast interval
            new_interval = data.get('interval', 30)
            if 5 <= new_interval <= 300:  # Between 5 seconds and 5 minutes
                self.broadcast_interval = new_interval
                await self.send_to_client(websocket, {
                    'type': 'interval_updated',
                    'new_interval': new_interval,
                    'status': 'success'
                })
                logger.info(f"Broadcast interval updated to {new_interval} seconds by client")
            else:
                await self.send_to_client(websocket, {
                    'type': 'error',
                    'message': 'Interval must be between 5 and 300 seconds'
                })
        elif message_type == 'client_info':
            # Client can send information about itself
            client_data = data.get('data', {})
            await self.send_to_client(websocket, {
                'type': 'client_info_received',
                'timestamp': time.time(),
                'client_address': str(websocket.remote_address),
                'status': 'acknowledged'
            })
            logger.info(f"Received client info: {client_data}")
        elif message_type == 'request_immediate_update':
            # Force an immediate status broadcast
            message = {
                'type': 'status_update',
                'timestamp': time.time(),
                'broadcast_reason': 'client_requested',
                **self.current_user_status
            }
            await self.send_to_client(websocket, message)
        elif message_type == 'server_ping':
            # Client requests server to ping them
            logger.info("Client requested server ping")
            await self.send_to_client(websocket, {
                'type': 'ping',
                'timestamp': time.time(),
                'message': 'Server ping - please respond with pong'
            })
        else:
            await self.send_to_client(websocket, {
                'type': 'error',
                'message': f'Unknown message type: {message_type}',
                'available_types': ['ping', 'pong', 'get_status', 'get_config', 'set_interval', 'client_info', 'request_immediate_update', 'server_ping']
            })
    
    async def interval_broadcast_task(self):
        """Background task to broadcast every 30 seconds when no user is detected"""
        while self.interval_broadcasting:
            try:
                await asyncio.sleep(self.broadcast_interval)
                if self.interval_broadcasting and not self.current_user_status['user_present']:
                    message = {
                        'type': 'status_update',
                        'timestamp': time.time(),
                        'broadcast_reason': 'interval',
                        **self.current_user_status
                    }
                    await self.broadcast_to_all(message)
                    logger.info("Interval broadcast sent (no user detected)")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in interval broadcast task: {e}")
    
    async def start_interval_broadcasting(self):
        """Start the interval broadcasting task"""
        if not self.interval_broadcasting:
            self.interval_broadcasting = True
            self.interval_task = asyncio.create_task(self.interval_broadcast_task())
            logger.info("Started interval broadcasting (30s intervals)")
    
    async def stop_interval_broadcasting(self):
        """Stop the interval broadcasting task"""
        if self.interval_broadcasting:
            self.interval_broadcasting = False
            if self.interval_task:
                self.interval_task.cancel()
                try:
                    await self.interval_task
                except asyncio.CancelledError:
                    pass
                self.interval_task = None
            logger.info("Stopped interval broadcasting")
    
    def update_user_status(self, user_present=False, user_count=0, distance=None, gender=None, age=None):
        """Update user status and broadcast to all clients"""
        current_time = time.time()
        
        # Check if user presence changed
        previous_user_present = self.current_user_status['user_present']
        
        # Update status
        self.current_user_status.update({
            'user_present': user_present,
            'user_count': user_count,
            'last_detection_time': current_time if user_present else self.current_user_status['last_detection_time'],
            'distance': distance,
            'gender': gender,
            'age': age
        })
        
        # Schedule broadcast and interval management in the event loop
        if self.loop and not self.loop.is_closed():
            try:
                # Handle broadcasting with throttling
                should_broadcast = False
                
                # Always broadcast when user presence changes
                if previous_user_present != user_present:
                    should_broadcast = True
                    self.last_immediate_broadcast = current_time
                    
                    if user_present:
                        # User detected: stop interval broadcasting
                        asyncio.run_coroutine_threadsafe(
                            self.stop_interval_broadcasting(),
                            self.loop
                        )
                        logger.info("User detected - stopped interval broadcasting")
                    else:
                        # User absent: start interval broadcasting
                        asyncio.run_coroutine_threadsafe(
                            self.start_interval_broadcasting(),
                            self.loop
                        )
                        logger.info("User absent - started interval broadcasting")
                
                # For ongoing presence updates, only throttle if user is present
                elif user_present and current_time - self.last_immediate_broadcast >= self.immediate_broadcast_throttle:
                    should_broadcast = True
                    self.last_immediate_broadcast = current_time
                # If user is absent, let the interval broadcasting handle it (don't broadcast here)
                
                # Send broadcast if needed
                if should_broadcast:
                    message = {
                        'type': 'status_update',
                        'timestamp': current_time,
                        'broadcast_reason': 'immediate',
                        **self.current_user_status
                    }
                    
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_to_all(message), 
                        self.loop
                    )
                
            except Exception as e:
                logger.error(f"Error scheduling broadcast: {e}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
        # Start interval broadcasting since initially no users are present
        await self.start_interval_broadcasting()
        
    def run_server(self):
        """Run the server in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.start_server())
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            if self.server:
                self.server.close()
            self.loop.close()
    
    def start(self):
        """Start the WebSocket server in a background thread"""
        self.thread = threading.Thread(target=self.run_server, daemon=True)
        self.thread.start()
        logger.info("WebSocket server thread started")
        
    def stop(self):
        """Stop the WebSocket server"""
        if self.loop and not self.loop.is_closed():
            # Stop interval broadcasting first
            if self.interval_broadcasting:
                asyncio.run_coroutine_threadsafe(
                    self.stop_interval_broadcasting(),
                    self.loop
                )
            self.loop.call_soon_threadsafe(self.loop.stop)
        logger.info("WebSocket server stopped")

# Global instance for easy access
websocket_server = None

def init_websocket_server(host='localhost', port=8765):
    """Initialize the global WebSocket server"""
    global websocket_server
    websocket_server = UserPresenceWebSocketServer(host, port)
    websocket_server.start()
    return websocket_server

def update_user_presence(user_present=False, user_count=0, distance=None, gender=None, age=None):
    """Update user presence status (to be called from main application)"""
    global websocket_server
    if websocket_server:
        websocket_server.update_user_status(
            user_present=user_present,
            user_count=user_count,
            distance=distance,
            gender=gender,
            age=age
        )

if __name__ == "__main__":
    # Test server
    server = init_websocket_server()
    
    try:
        # Simulate user presence updates
        time.sleep(2)
        update_user_presence(True, 1, 1.5, 'M', 25)
        time.sleep(5)
        update_user_presence(False, 0)
        time.sleep(5)
        update_user_presence(True, 1, 2.0, 'F', 30)
        
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()