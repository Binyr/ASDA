from multiprocessing import Pool, Process, Queue, Pipe
import numpy as np 



def write(q, l):
	q.put(100)
	l.append(99)

def write2(conn):
	a = conn.recv()
	a[0]=100
	print(a)
	print(id(a))
	conn.send(a)

if __name__ == '__main__':
	a = np.ones((5))
	q = Queue()
	l = []

	print(q)
	print(l)
	pw = Process(target=write, args=(q,l))
	pw.start()
	pw.join()
	print(q)
	print(l)

	conn1, conn2 = Pipe()
	pw2 = Process(target=write2, args=(conn2,))
	pw2.start()
	conn1.send(a)
	pw2.join()
	print(a)
	print(id(a))
	print(id(conn1.recv()))


