import socket
import sys

host = ""
port = 1337

localIP = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)),
s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET,
socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

s.bind((host, port))
s.listen(1)

print("[#] You ip is {0}".format(localIP))
print("[#] Port: {0}".format(port))
link = input("Input the link to redirect to: ")

if(not link.startswith("http")):
    link = "http://" + link

print("[#] Waiting for connection")
sock, addr = s.accept()

file = sock.makefile("rw", buffering=1)

line = file.readline().strip()

print("[#] Sending request")
file.write('HTTP/1.0 200 OK\n\n')
file.write("<html><head><script>window.location.replace('{0}')</script></head>".format(link))
file.write("<body><h1>Redirecting to {0}</body></html>".format(link))

print("[#] Done sending. Quiting")
