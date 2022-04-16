These instructions will guide you through running the experiments presented in the Virtual Neuron paper (SC22 ID: pap433).

1. Open a terminal window and open the docker image for the NEST neuromorphic simulator:
	docker run --rm -it -e LOCAL_USER_ID=`id -u $USER` -v $(pwd):/opt/data -p 8080:8080 nestsim/nest:3.3 /bin/bash

2. In the docker container, navigate to the root:
	cd root

3. In a separate terminal window, clone the virtual neuron GitHub repository and navigate to it:
	git clone https://github.com/prasannadate/virtual-neuron.git 
	cd virtual-neuron

4. Use 'docker cp' command to copy the unzipped files to the root directory in the docker container:
	docker cp VirtualNeuron.py <docker-container-id>:/root/
	docker cp test_8_bit.py <docker-container-id>:/root/
	docker cp test_16_bit.py <docker-container-id>:/root/
	docker cp test_32_bit.py <docker-container-id>:/root/
	docker cp test_constant_function.py <docker-container-id>:/root/
	docker cp test_successor_function.py <docker-container-id>:/root/
	docker cp test_predecessor_function.py <docker-container-id>:/root/
	docker cp test_multiply_neg_one.py <docker-container-id>:/root/

5. Check if the test files have been copied into root of the docker container by doing a 'ls' in the docker container:
	ls

6. Run the test files in the docker container
	python test_8_bit.py
	python test_16_bit.py
	python test_32_bit.py
	python test_constant_function.py
	python test_successor_function.py
	python test_predecessor_function.py
	python test_multiply_neg_one.py

7. During each run, check to see if "Correct Results!" appears on the program output. This indicates that the neuromorphic algorithm is running correctly!
