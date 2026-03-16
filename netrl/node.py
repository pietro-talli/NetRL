# A node is a generic unit which can communicate with the central server
class Node:
    def send(self, data):
        """
        Send data to the central server.
        Parameters:
        data: The data to send.
        """
        raise NotImplementedError("send method not implemented")

    def receive(self):
        """
        Receive data from the central server.
        Returns:
        The data received from the central server.
        """
        raise NotImplementedError("receive method not implemented")