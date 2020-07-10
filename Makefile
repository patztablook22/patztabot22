all:
	mvn package -q
	java -jar target/patztabot22-1.jar #token
