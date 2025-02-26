from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import paho.mqtt.client as mqtt
import numpy as np
from scipy import signal
import cbor2  # For decoding CBOR data
import logging
import json  # For JSON serialization

# Flask App Configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://sy:sleeps@localhost/sleep_monitor'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# MQTT Configuration
MQTT_BROKER = "192.168.1.117"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/data"

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Buffer for 30 Seconds (200 Hz * 30 seconds = 6000 samples)
DATA_BUFFER_SIZE = 6000
data_buffer = []

# Database Model
class SleepData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    raw_data = db.Column(db.JSON, nullable=False)
    breath_rate = db.Column(db.Float)
    heart_rate = db.Column(db.Float)

# MQTT Client Setup
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    """ Callback when the MQTT client connects to the broker """
    if rc == 0:
        logger.info("Connected to MQTT Broker")
        client.subscribe(MQTT_TOPIC)
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code: {rc}")

def on_message(client, userdata, msg):
    """ Callback when a message is received from the MQTT broker """
    with app.app_context():  # Activate Flask application context
        try:
            logger.info(f"Received message from topic: {msg.topic}")

            # Log the raw payload (for debugging)
            logger.info(f"Raw payload (hex): {msg.payload.hex()}")

            # Decode CBOR data
            data = cbor2.loads(msg.payload)  # Decode CBOR payload
            logger.info(f"Decoded CBOR data: {data}")

            if 'data' not in data:
                logger.error("Invalid CBOR format: 'data' field missing")
                return

            # Append new data to the buffer
            global data_buffer
            data_buffer.extend(data['data'])

            # If buffer has enough data (6000 samples), analyze it
            if len(data_buffer) >= DATA_BUFFER_SIZE:
                # Convert buffer to NumPy array
                raw_data = np.array(data_buffer[:DATA_BUFFER_SIZE], dtype=float)

                # Analyze data to extract breath and heart rates
                breath_rate, heart_rate = analyze_data(raw_data)

                # Convert numpy.float64 to Python float
                breath_rate = float(breath_rate) if breath_rate is not None else None
                heart_rate = float(heart_rate) if heart_rate is not None else None

                # Serialize raw_data as JSON
                raw_data_json = json.dumps(data_buffer[:DATA_BUFFER_SIZE])

                # Store data in the database
                new_record = SleepData(
                    timestamp=datetime.now(),
                    raw_data=raw_data_json,  # Store raw data as JSON
                    breath_rate=breath_rate,
                    heart_rate=heart_rate
                )
                db.session.add(new_record)
                db.session.commit()

                logger.info("Data logged successfully")

                # Clear the buffer
                data_buffer = data_buffer[DATA_BUFFER_SIZE:]

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            db.session.rollback()

def analyze_data(sensor_values, fs=200):
    """ Analyze sensor data to extract breath and heart rate """
    try:
        detrended_data = signal.detrend(sensor_values)
        sos_prefilter = signal.butter(10, 5.0, btype='low', fs=fs, output='sos')
        filtered_data = signal.sosfilt(sos_prefilter, detrended_data)
        
        f, Pxx = signal.welch(filtered_data, fs, nperseg=fs*30, window='hann', scaling='density')
        
        def find_highest_peak(f, Pxx, freq_range):
            """ Find highest peak in given frequency range """
            mask = (f >= freq_range[0]) & (f <= freq_range[1])
            if np.any(mask):  # Ensure mask is not empty
                peak_idx = np.argmax(Pxx[mask])
                return f[mask][peak_idx]
            return 0  # Return 0 instead of None to avoid NoneType errors

        breath_freq = find_highest_peak(f, Pxx, (0.1, 0.5))
        heart_freq = find_highest_peak(f, Pxx, (0.7, 3))

        return breath_freq * 60, heart_freq * 60  # Convert Hz to BPM

    except Exception as e:
        logger.error(f"Data analysis error: {e}")
        return None, None

# Flask Routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/data', methods=['GET'])
def fetch_data():
    """ Fetch the latest sleep data from the database """
    try:
        latest_entry = SleepData.query.order_by(SleepData.timestamp.desc()).first()
        if latest_entry:
            return jsonify({
                'timestamp': latest_entry.timestamp.isoformat(),
                'breath_rate': latest_entry.breath_rate,
                'heart_rate': latest_entry.heart_rate,
                'raw_data': latest_entry.raw_data  # Include raw sensor data
            })
        return jsonify({'message': 'No data available'}), 404
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return jsonify({'error': str(e)}), 500

# Check Database Connection
def check_database_connection():
    """ Verify that the database is connected """
    try:
        db.engine.connect()
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

# Start MQTT Client
def start_mqtt_client():
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()

# Main Function
if __name__ == '__main__':
    with app.app_context():
        try:
            # Check database connection before starting the app
            check_database_connection()
            db.create_all()  # Create database tables if they don't exist
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            exit(1)

    start_mqtt_client()  # Start the MQTT client
    app.run(host='0.0.0.0', port=5000)