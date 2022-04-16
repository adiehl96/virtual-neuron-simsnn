import numpy as np 
import time 
import nest 
from VirtualNeuron import VirtualNeuron, connect_virtual_neurons




# Function to get binary number as a list from recorder
def recorders_to_binary(recorders):
	binary = []

	for i in range(len(recorders)):
		events = recorders[i].get("events")

		if events["times"].size > 0:
			binary.append(1)
		else:
			binary.append(0)

	binary.reverse()

	return np.array(binary)


# Function to get positive decimal number from binary array
def binary_to_decimal_positive(binary):
	powers_of_two = []

	for i in range(precision[0]):
		powers_of_two.append(2**(precision[0] - i - 1))

	for i in range(precision[1]):
		powers_of_two.append(2**(-i-1))

	if len(binary) > len(powers_of_two):
		powers_of_two.insert(0, powers_of_two[0]*2)

	return binary @ powers_of_two


# Function to get negative decimal number from binary array
def binary_to_decimal_negative(binary):
	powers_of_two = []

	for i in range(precision[2]):
		powers_of_two.append(-2**(precision[0] - i - 1))

	for i in range(precision[3]):
		powers_of_two.append(-2**(-i-1))

	if len(binary) > len(powers_of_two):
		powers_of_two.insert(0, powers_of_two[0]*2)

	return binary @ powers_of_two




# Precision 
precision = [8,0,8,0] 	# 16-bit precision
positive_precision = precision[0] + precision[1]
negative_precision = precision[2] + precision[3]

print("Total Precision:", precision)
print("Positive Precision:", positive_precision, "bits")
print("Negative Precision:", negative_precision, "bits")


# Constant
constant_pos = np.array([0]*(positive_precision-1) + [1])
constant_neg = np.zeros(negative_precision)

print("\nPositive Constant:", constant_pos)
print("Negative Constant:", constant_neg)


# Start timer
start_time = time.time()


# Total number of runs
num_runs = 100000



# Start runs
for t in range(num_runs):
	print("\nRun", t+1, "/", num_runs)

	# Random numbers
	number = np.random.randint(2, size=(positive_precision + negative_precision))

	# A and B for this iteration
	a_pos = constant_pos
	a_neg = constant_neg
	b_pos = number[0 : positive_precision]
	b_neg = number[positive_precision : ]

	# print("a_pos:", a_pos)
	# print("b_pos:", b_pos)
	# print("a_neg:", a_neg)
	# print("b_neg:", b_neg)


	# Create virtual neurons
	A = VirtualNeuron(precision)
	B = VirtualNeuron(precision)
	C = VirtualNeuron(precision)


	# Connect virtual neurons
	for i in range(A.positive_precision):
		nest.Connect(A.z_positive[i], C.x_positive[i], syn_spec={"weight": 1.0, "delay": 1.0})

	for i in range(A.negative_precision):
		nest.Connect(A.z_negative[i], C.x_negative[i], syn_spec={"weight": 1.0, "delay": 1.0})

	for i in range(B.positive_precision):
		nest.Connect(B.z_positive[i], C.y_positive[i], syn_spec={"weight": 1.0, "delay": 1.0})

	for i in range(B.negative_precision):
		nest.Connect(B.z_negative[i], C.y_negative[i], syn_spec={"weight": 1.0, "delay": 1.0})




	# Create spike generator 
	spike_generator = nest.Create("spike_generator")
	spike_generator.set({"spike_times": [1.0]})


	# Connect spike generator to neurons
	default_syn_spec = {"weight": 1.0, "delay": 1.0}

	for i in range(positive_precision):
		if a_pos[positive_precision - i - 1] == 1:
			nest.Connect(spike_generator, A.x_positive[i], syn_spec=default_syn_spec)

		if b_pos[positive_precision - i - 1] == 1:
			nest.Connect(spike_generator, B.x_positive[i], syn_spec=default_syn_spec)

	for i in range(negative_precision):
		if a_neg[negative_precision - i - 1] == 1:
			nest.Connect(spike_generator, A.x_negative[i], syn_spec=default_syn_spec)

		if b_neg[negative_precision - i - 1] == 1:
			nest.Connect(spike_generator, B.x_negative[i], syn_spec=default_syn_spec)


	# Create spike recorder 
	A_rec_pos = {}
	B_rec_pos = {}
	C_rec_pos = {}

	A_rec_neg = {}
	B_rec_neg = {}
	C_rec_neg = {}

	if positive_precision > 0:
		for i in range(positive_precision + 1):
			A_rec_pos[i] = nest.Create("spike_recorder")
			B_rec_pos[i] = nest.Create("spike_recorder")
			C_rec_pos[i] = nest.Create("spike_recorder")
			
			nest.Connect(A.z_positive[i], A_rec_pos[i])
			nest.Connect(B.z_positive[i], B_rec_pos[i])
			nest.Connect(C.z_positive[i], C_rec_pos[i])


	if negative_precision > 0:
		for i in range(negative_precision + 1):
			A_rec_neg[i] = nest.Create("spike_recorder")
			B_rec_neg[i] = nest.Create("spike_recorder")
			C_rec_neg[i] = nest.Create("spike_recorder")
			
			nest.Connect(A.z_negative[i], A_rec_neg[i])
			nest.Connect(B.z_negative[i], B_rec_neg[i])
			nest.Connect(C.z_negative[i], C_rec_neg[i])



	# Simulate
	nest.Simulate((positive_precision + negative_precision)*2)


	# Display positive spike recorders
	print("\nPositive Results:")

	if positive_precision > 0:
		# print("\nA:")
		# for i in range(positive_precision + 1):
		# 	events = A_rec_pos[positive_precision - i].get("events")
		# 	print(events)

		# print("\nB:")
		# for i in range(positive_precision + 1):
		# 	events = B_rec_pos[positive_precision - i].get("events")
		# 	print(events)

		# print("\nC:")
		# for i in range(positive_precision + 1):
		# 	events = C_rec_pos[positive_precision - i].get("events")
		# 	print(events)

		a_pos_dec = binary_to_decimal_positive(a_pos)
		b_pos_dec = binary_to_decimal_positive(b_pos)
		c_pos_dec = binary_to_decimal_positive(recorders_to_binary(C_rec_pos))

		print("A:", a_pos_dec)
		print("B:", b_pos_dec)
		print("C:", c_pos_dec)



	print("\nNegative Results:")

	if negative_precision > 0:
		# print("\nA:")
		# for i in range(negative_precision + 1):
		# 	events = A_rec_neg[negative_precision - i].get("events")
		# 	print(events)

		# print("\nB:")
		# for i in range(negative_precision + 1):
		# 	events = B_rec_neg[negative_precision - i].get("events")
		# 	print(events)

		# print("\nC:")
		# for i in range(negative_precision + 1):
		# 	events = C_rec_neg[negative_precision - i].get("events")
		# 	print(events)

		a_neg_dec = binary_to_decimal_negative(a_neg)
		b_neg_dec = binary_to_decimal_negative(b_neg)
		c_neg_dec = binary_to_decimal_negative(recorders_to_binary(C_rec_neg))

		print("A:", a_neg_dec)
		print("B:", b_neg_dec)
		print("C:", c_neg_dec)

	if (c_pos_dec == b_pos_dec + 1):
		print("\nCorrect Results!")
	else:
		print("\nIncorrect Results!")
		break



	# Reset Kernel
	nest.ResetKernel()


	print("\n\n\n")


# End time
end_time = time.time()

print("Total time elapsed:", (end_time - start_time)/60.0, "minutes")











































