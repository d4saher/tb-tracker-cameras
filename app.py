import numpy as np
import cv2 as cv
import time
import paho.mqtt.client as mqtt
import json
import os
import configparser

from camera import Camera

config = configparser.ConfigParser()
config_file = "/home/testbed/tb-tracker/tb-tracker-cameras/camera_config.ini"

if os.path.exists(config_file):
    config.read(config_file)
else:
    print(f"Config file {config_file} doesn't exist.")

CAMERA_ID = config.get("camera", "id", fallback=None)

print(f"Camera ID: {CAMERA_ID}")

client = mqtt.Client()

is_capturing_points = False

dev = False

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

    client.subscribe("tb-tracker/is_capturing_points")
    client.subscribe("tb-tracker/set_gain")
    client.subscribe("tb-tracker/set_exposure")

def on_message(client, userdata, msg):
    global is_capturing_points
    camera = Camera.instance()
    if msg.topic == "tb-tracker/is_capturing_points" and json.loads(msg.payload.decode()) != is_capturing_points:
        is_capturing_points = json.loads(msg.payload.decode())
        camera.set_is_capturing_points(is_capturing_points)
    elif msg.topic == "tb-tracker/set_gain":
        gain = json.loads(msg.payload.decode())
        gain = int(np.interp(gain, [0, 100], [0, 63]))
        camera.set_gain(gain)
    elif msg.topic == "tb-tracker/set_exposure":
        exposure = json.loads(msg.payload.decode())
        camera.set_exposure(exposure)

def send_points(points, timestamp):
    
    if points == []:
        points = [[None, None]]

    #print(f"Sending points: {points}")
    topic = f"tb-tracker/{CAMERA_ID}/points"
    #print(f"Topic: {topic}")
    payload = {
        "timestamp": timestamp,
        "points": points
    }

    client.publish(topic, json.dumps(payload))

if __name__ == "__main__":
    camera = Camera.instance()
    
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect("172.16.0.102", 1883, 60)

    client.loop_start()
    while True:
        frame, image_points, timestamp = camera.get_frame()
        #print(image_points)
        
        send_points(image_points, timestamp)

        if dev:
            cv.imshow("Camera", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

