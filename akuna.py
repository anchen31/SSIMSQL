

# better to use regex

# check if it is a 8 digit number, if it is not then pass


import re
import numpy as np
import math

def validate(value):
	if len(value) == 8:
		holder = 0
		first_part = str(int(value[0:6], 16))

		for char in first_part:
			holder += int(char)

		first_part = hex(holder)
		Second_part = hex(int(value[6:], 16))

		if (first_part == Second_part):
			print('VALID')
		else:
			print("INVALID")

	else:
		print('INVALID')


# validate('C0FFEE1C')
# validate('8BADF00D')



######################################################################

# input is a list of strings

def travelDistance(locTrip):

	array = np.empty(shape=(0,3))
	values = []
	# parse the string to get the info, use regex:
	for i in locTrip:
		x = re.split(':', i)
		# check if it is a location or trip:
		if x[0] == 'LOC':
			array = np.append(array, [x[1:]], axis=0)
		elif x[0] == 'TRIP':
			d1 = x[2]
			d2 = x[3]
			# check if these two are in the matrix
			for j in array:
				if d1 or d2 in j:
					values.append(j[1].astype(float))
					values.append(j[2].astype(float))



			location = math.acos((math.sin(math.radians(values[0])) * math.sin(math.radians(values[2])))+(math.cos(math.radians(values[1]))*math.cos(math.radians(values[3]))*math.cos(abs(values[1]-values[3]))))
			print(location)
		else:
			pass


travelDistance(['LOC:CHI:41.836944:-87.684722', 'LOC:NYC:40.7127:-74.0059', 'TRIP:C0FFEE1C:CHI:NYC'])