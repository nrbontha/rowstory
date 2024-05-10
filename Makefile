build:
	docker build -t mypostgresdb .

run: build
	docker run -d --name mypostgrescontainer -p 5432:5432 mypostgresdb

stop:
	docker stop mypostgrescontainer

clean:
	docker rm mypostgrescontainer
	docker rmi mypostgresdb

.PHONY: build run stop clean
