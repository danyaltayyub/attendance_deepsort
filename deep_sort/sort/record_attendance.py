from typing import Any
import requests
import csv
from datetime import datetime


class RecordAttendance():
    def __init__(self):
        self.attendance_file = "/home/transdata/DeepSORT_Face/att.csv"
        self.employee_ids = "/home/transdata/DeepSORT_Face/employee_ids.csv"

        self.api_url = "https://app.inteliviu.com/dev/api/mark-attendance"

    def get_employee_id(self, name):
        """
        Get the employee ID from the CSV file based on the employee name.
        """
        with open(self.employee_ids, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == name:
                    return row[1]
        return None
    
    def store_attendance(self, name):
        """
        Store the attendance entry in the CSV file.
        """
        employee_id = self.get_employee_id(name)

        if employee_id is None:
            print("Employee ID not found")
            return
        
        datetime_var = datetime.now()
        current_date = datetime_var.strftime('%Y-%m-%d')
        current_time = datetime_var.strftime('%H:%M:%S')
        # time_int = self.get_time_int()
        
        # Check for duplicate entry
        if self.check_duplicate(name, current_date, current_time):
            print("Duplicate entry found. Skipping.")
            return



        # Store attendance entry
        with open(self.attendance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, employee_id, current_date, current_time])
            print("Attendance entry stored successfully.")
        
        
        # Post attendance on portal
        self.post_attend(employee_id, datetime_var.strftime("%Y-%m-%d %H:%M:%S"))
    
    def check_duplicate(self, name, cur_date, cur_time):
        """
        Check if a name already exists in the CSV file.
        """
        with open(self.attendance_file, mode='r') as file:
            reader = csv.reader(file)
            check = False
            for row in reader:
                if row[0] == name:
                    if row [2] == cur_date:
                        if (self.get_time_int(row[3]) - self.get_time_int(cur_time)) < 300:
                            check = True
                        else :
                            check = False
                
        return check

    def get_time_int(self , time_str):
        time_format = "%H:%M:%S"
        time_object = datetime.strptime(time_str , time_format)
        # time_int = datetime.now().time()
        time_abs = (time_object.hour * 3600) + (time_object.minute * 60) + (time_object.second)
        return time_abs
    
    def post_attend(self, emp_id , date_time):
        print(date_time)
        data = {
                    "employee_id": emp_id,  # Replace with actual Employee ID
                    "clock_in": date_time,  # Replace with actual datetime
                }

        # Sending the POST request with JSON data
        response = requests.post(self.api_url, json=data)
        response_data = response.content.decode('utf-8')  # Decode the response content
        

        # Checking the response
        if response.status_code == 200:
            print("Response:", response_data)
            # json_load = json.loads(response)
            # print(json_load)
            print("Attendance marked successfully!")
        else:
            print("Response:", response_data)
            print("Failed to mark attendance. Status code:", response.status_code)
