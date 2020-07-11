all: build run

build:
	mvn package -q

run:
	java -jar target/patztabot22-1.jar #token_here

