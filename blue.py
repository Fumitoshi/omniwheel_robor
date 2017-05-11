# -*- coding: utf-8 -*-
import bluetooth
import struct
import time
import sys

target_name1 = "TOSABT1-001bdc064f4e"
target_address1 = None
target_name2 = "TOSABT2-001bdc063236"
target_address2 = None

##nearby_devices = bluetooth.discover_devices()
##
##for bdaddr1 in nearby_devices:
##    if target_name1 == bluetooth.lookup_name( bdaddr1 ):
##        target_address1 = bdaddr1
##        break
##for bdaddr2 in nearby_devices:
##    if target_name2 == bluetooth.lookup_name( bdaddr2 ):
##        target_address2 = bdaddr2
##        break
##if target_address1 is not None:
##    print "found target bluetooth device with address ", target_address1
##else:
##    print "could not find target bluetooth device nearby"
##if target_address2 is not None:
##    print "found target bluetooth device with address ", target_address2
##else:
##    print "could not find target bluetooth device nearby"
##

bd_addr1 = "00:1b:dc:06:4f:4e"
bd_addr2 = "00:1b:dc:06:32:36"

port = 1

s1=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
##s2=bluetooth.BluetoothSocket(bluetooth.RFCOMM)

print "0"
s1.connect((bd_addr1, port))
print"5"
##s2.connect((bd_addr2, port))
print "1"
print"2"
s1.send("a" "200" "200" "200" "200" "Chr(13)" )
print"100"
s1.close()
##s2.close()