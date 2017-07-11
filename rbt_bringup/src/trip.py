#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from math import pi
from  std_msgs.msg import String


class Trip():
    def __init__(self):
        rospy.init_node('trip', anonymous=False)
        rospy.on_shutdown(self.shutdown)
  
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rate = 50
        r = rospy.Rate(rate)

        linear_vel = 0.4
        goal_linear = 1.0
        linear_duration = goal_linear / linear_vel
      
        angular_vel = 0.4
        goal_angular= pi
        angular_duration = goal_angular / angular_vel
      
        for i in range(2):

            self.move_cmd = Twist()
            self.move_cmd.linear.x =  linear_vel 

            ticks = int (linear_duration * rate)
            
            for t in range(ticks):
                self.cmd_vel_pub.publish(self.move_cmd)
                r.sleep()

            self.move_cmd = Twist()
            self.cmd_vel_pub.publish(self.move_cmd)
            rospy.sleep(2.0)

            self.move_cmd.angular.z =  angular_vel
            
            ticks = int (angular_duration * rate)
                 
            for t in range(ticks):
                self.cmd_vel_pub.publish(self.move_cmd)
                r.sleep()
            
            self.move_cmd = Twist()
            self.cmd_vel_pub.publish(self.move_cmd)
            rospy.sleep(2.0)
        
        self.cmd_vel_pub.publish(self.move_cmd)
           
        

    def shutdown():
        rospy.loginfo('Trip over...')
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1.0)        
      
        
if __name__ == '__main__':
    try:
        Trip()
    except :
        rospy.loginfo('Node terminated')