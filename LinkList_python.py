class node:
    def __init__(self,data):
        self.data=data
        self.next=None
class linklist:
    def __init__(self):
        self.head=None
    def beg(self,newdata):
        newnode=node(newdata)
        newnode.next=self.head
        self.head=newnode
    def last(self,newdata):
        newnode=node(newdata)
        if self.head is None:
            self.head=newnode
        temp=self.head
        while temp.next is not None:
            temp=temp.next
        temp.next=newnode
    def insertatposition(self,newdata,pos):
        newnode=node(newdata)
        position=0
        temp=self.head
        while temp is not None:
            if position==pos:
                previousnode.next=newnode
                newnode.next=temp
            previousnode=temp
            temp=temp.next
            position+=1
    def printlist(self):
        temp=self.head
        while temp is not None:
            print(temp.data)
            temp=temp.next
l=linklist()
l.head=node(1)
second=node(2)
l.head.next=second
l.beg(3)
l.last(4)
l.insertatposition(6,3)
l.printlist()
