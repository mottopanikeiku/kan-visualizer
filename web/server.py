#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8002

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    # Change to the directory containing this script
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🚀 KAN Visualizer server starting...")
        print(f"📁 Serving from: {os.getcwd()}")
        print(f"🌐 Open your browser to: http://localhost:{PORT}")
        print(f"⚠️  Press Ctrl+C to stop the server")
        
        try:
            # Try to open the browser automatically
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
        
        httpd.serve_forever()

if __name__ == "__main__":
    start_server() 