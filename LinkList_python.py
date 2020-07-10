class node:
    def __init__(self,data):
        self.data=data
        self.next=None
class linklist:
    def __init__(self):
        self.head=None
    def traversal(self):
        temp=self.head
        while temp is not None:
            print(temp.data)
            temp=temp.next
    def beginning(self,newdata):
        newnode=node(newdata)
        newnode.next=self.head
        self.head=newnode
    def last(self,newdata):
        newnode=node(newdata)
        while self.head is None:
            self.head=newnode
        temp=self.head
        while(temp.next):
            temp=temp.next
        temp.next=newnode
l=linklist()
l.head=node(1)
second=node(2)
third=node(3)
l.head.next=second
second.next=third
l.beginning(0)
l.last(4)
l.traversal()
            



























