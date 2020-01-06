# Server
import http.server
from http.server import BaseHTTPRequestHandler, HTTPServer
# import socketserver

# Controller
from pyautogui import press, keyDown, keyUp

keys = [
    {
        "left": "q",
        "right": "w",
        "down": "e",
        "up": "r",
        "interact": ["t", "y"],
        "jump": "u"
    },
    {
        "left": "i",
        "right": "o",
        "down": "p",
        "up": "[",
        "interact": ["]", "n"],
        "jump": "a"
    },
    {
        "left": "s",
        "right": "d",
        "down": "f",
        "up": "g",
        "interact": ["h", "j"],
        "jump": "k"
    }
]

def changeDirection(player, dir, b):
    print(keys[player][dir])
    if(b == "true"):
        if(dir == "left"):
            keyDown(keys[player][dir])
        elif(dir == "right"):
            keyDown(keys[player][dir])
        elif(dir == "up"):
            keyDown(keys[player][dir])
        elif(dir == "down"):
            keyDown(keys[player][dir])
    elif(b == "false"):
        if(dir == "left"):
            keyUp(keys[player][dir])
        elif(dir == "right"):
            keyUp(keys[player][dir])
        elif(dir == "up"):
            keyUp(keys[player][dir])
        elif(dir == "down"):
            keyUp(keys[player][dir])
    else:
        if(dir == "jump"):
            press(keys[player][dir])
        if(dir == "interact"):
            press(keys[player][dir][0])
            press(keys[player][dir][1])

HOST = ""
PORT = 8080

playerIndex = 0
class myHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global playerIndex

        if("/reset" in self.path):
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.send_header("Set-Cookie", "player=99")
            self.end_headers()
            return

        if("/changeDirection" in self.path):
            dir = self.path.split("dir=")[1].split("&")[0]
            b = self.path.split("b=")[1]
            index = int(str(self.headers).split("Cookie: player=")[1].split("\n")[0])
            changeDirection(index, dir, b)

            self.send_response(204)
            self.end_headers()
            return

        mime_type = ""

        if(self.path.endswith(".html")):
            mime_type = "text/html"
        elif(self.path.endswith(".js")):
            mime_type = "application/javascript"
        elif(self.path.endswith(".css")):
            mime_type = "text/css"
        elif(self.path.endswith(".ico")):
            self.send_error(404, "Dont have an icon")
            return

        self.send_response(200)
        self.send_header("Content-type", mime_type)

        cookie = str(self.headers).split("Cookie: player=")
        if(len(cookie) == 1):
            self.send_header("Set-Cookie", "player=" + str(playerIndex))
            print("New player {}".format(playerIndex))
            playerIndex += 1
        else:
            cookie = int(cookie[1].split("\n")[0])
            if(cookie >= playerIndex):
                self.send_header("Set-Cookie", "player=" + str(playerIndex))
                print("Reassigning player to {}".format(playerIndex))
                playerIndex += 1

        self.end_headers()

        file = self.path[1:]
        if(file == ""):
            file = "index.html"

        try:
            f = open(file, "rb")
            self.wfile.write(f.read())
            f.close()
        except Exception:
            self.send_error(404, "File not found")
        return


# handler = http.server.SimpleHTTPRequestHandler
# handler = Server()

try:
    server = HTTPServer((HOST, PORT), myHandler)
    print("Listening on port", PORT)
    server.serve_forever()
except KeyboardInterrupt:
    print("Closing server")
    server.socket.close()