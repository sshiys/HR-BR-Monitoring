## How to run this project
1. Create a virtual environment
```
python -m venv virtualname
```
2. Activate virtual environment
```
virtualname\Scripts\activate
```
3. To install libraries
```
pip install requirements.txt
```

***or***

```
pip install flask flask-sqlalchemy psycopg2-binary numpy scipy plotly requests paho-mqtt
```
4. Set postgres database
```
psql -U postgres
```
enter your credential
**set up the database**
```
CREATE USER <<username>> WITH ENCRYPTED PASSWORD '<<password>>';
```
CREATE DATABASE sleep_monitor;
GRANT ALL ON DATABASE sleep_monitor TO <<nama user>>;
ALTER DATABASE sleep_monitor OWNER TO <<nama user>>;
exit : ctrl c
*user sy, pass:sleeps
check queries
psql -U sy -d sleep_monitor
SELECT * FROM sleep_data; 
5. To run Flask Server
```
Python server.py
```
6. To run dashboard (in another terminal)
```
streamlit run dashboard.py
```
Install Mosquitto
And run the installer
add or edit mosquitto configuration file "mosquitto.conf"
add these lines

open cmd with administrator right
move to the mosquitto directory
```
cd "C:/program files/mosquitto"
```
run mosquitto with configuration
mosquitto -c mosquitto.conf -v

if you are not sure what's your IP address: check using
```
ipconfig
```