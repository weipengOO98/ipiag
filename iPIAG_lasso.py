import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rc


class Event:
    # event_type includes
    # "W_rec": worker i receives from Master
    # "W_cpt": worker i complete job
    # "B_rec": buffer receives from worker i
    # "M_cpt": master complete job
    def __init__(self, clock=0., event_type="-1",
                 data=None):
        self.clock = clock
        self.event_type = event_type
        if data is None:
            self.data = {"index": -1, "k_iter": 0, "coordinates": np.array(0.), "gradient": np.array(0.)}
        else:
            self.data = data.copy()

    def display(self):
        print("At %.2f, %s happens." % (self.clock, self.event_type))


class Schedule:
    def __init__(self):
        self.queue = []

    def enqueue(self, new_event):
        for i in range(len(self.queue)):
            if new_event.clock < self.queue[i].clock:
                self.queue.insert(i, new_event)
                return 0
        self.queue.append(new_event)
        return 0

    def dequeue(self):
        if self.is_empty():
            print("Nothing will be done.")
            return -1
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0

    def display(self):
        for event in self.queue:
            print("%.2f: %s" % (event.clock, event.event_type))


class Worker:
    def __init__(self, index=0, rcvDelay=3., rcvStd=0.01, procDelay=3., procStd=0.01, sndDelay=3, sndStd=0.01,
                 func=lambda x: x):
        self.index = index
        self.rcvDelay = rcvDelay
        self.rcvStd = rcvStd
        self.procDelay = procDelay
        self.procStd = procStd
        self.sndDelay = sndDelay
        self.sndStd = sndStd
        self.completeTime = stats.truncnorm((0 - self.procDelay) / self.procStd, (10 * self.procDelay
                                                                                  - self.procDelay) / self.procStd,
                                            loc=self.procDelay, scale=self.procStd)
        self.sndTime = stats.truncnorm((0 - self.sndDelay) / self.sndStd, (10 * self.sndDelay
                                                                           - self.sndDelay) / self.sndStd,
                                       loc=self.sndDelay, scale=self.sndStd)
        self.rcvTime = stats.truncnorm((0 - self.rcvDelay) / self.rcvStd, (10 * self.rcvDelay
                                                                           - self.rcvDelay) / self.rcvStd,
                                       loc=self.rcvDelay, scale=self.rcvStd)
        self.xdata = {"coordinates": np.array(0.)}
        self.gdata = {"gradient": np.array(0.)}
        self.func = func

    def update(self):
        self.gdata["gradient"] = self.func(self.xdata["coordinates"])

    def generateCompleteTime(self):
        return self.completeTime.rvs(1)[0]

    def generateReceiveTime(self):
        return self.rcvTime.rvs(1)[0]

    def generateSendTime(self):
        return self.sndTime.rvs(1)[0]


class Buffer:
    def __init__(self):
        self.empty = True
        # list(event_data)
        self.datalist = []
        # list(int)
        self.sendlist = []

    def flush(self):
        self.datalist = []
        self.sendlist = []

    def receive(self, data):
        self.datalist.append(data)
        self.sendlist.append(data["index"])

    def is_empty(self):
        if len(self.sendlist) == 0:
            return True
        else:
            return False


class Master:
    def __init__(self, procDelay=10., procStd=0.01, initx=lambda x: x, updatex=lambda x: x,bufferthresh=1):
        self.procDelay = procDelay
        self.procStd = procStd
        self.completeTime = stats.truncnorm((0 - self.procDelay) / self.procStd,
                                            (10 * self.procDelay - self.procDelay) / self.procStd, loc=self.procDelay,
                                            scale=self.procStd)
        self.k_iter = 0
        self.buffer = Buffer()
        self.maxiter = 10000
        # list(int)
        self.sendlist = []
        # list(event_data)
        self.is_idle = True
        self.xdata = dict()
        self.updatex = updatex
        self.bufferthresh = bufferthresh
        initx(self)

    def generateCompleteTime(self):
        return self.completeTime.rvs(1)[0]

    def initialization(self):
        return self.k_iter

    def getKiter(self):
        return self.k_iter

    def fetch_data(self):
        for event_data in self.buffer.datalist:
            self.xdata["gradients"][event_data["index"]] = event_data["gradient"]
        self.sendlist = self.buffer.sendlist.copy()
        self.buffer.flush()

    def update(self):
        # sum_gradient = 0
        # for gradient in self.xdata["gradients"]:
        #    sum_gradient +=gradient
        # self.updatedx["coordinates"] -= 0.0002*sum_gradient
        self.updatex(self)


class ScheduleDriver:
    def __init__(self, worker_list, master):
        self.schedule = Schedule()
        self.worker_list = worker_list
        self.master = master
        self.currentTime = 0
        self.masterResultList = []
        for i in range(len(worker_list)):
            worker = worker_list[i]
            # print("%.2f: Send initials to worker %d" % (self.currentTime, worker.index))
            new_event = Event(worker.generateReceiveTime() + self.currentTime,
                              event_type="W_rec",
                              data={"index": i, "k_iter": 0, "coordinates": self.master.updatedx["coordinates"].copy()})
            self.schedule.enqueue(new_event)

    def next_event(self):
        if self.schedule.is_empty():
            print("Something goes wrong, nothing to do!!!")
            return -1
        event = self.schedule.dequeue()
        self.currentTime = event.clock
        try:
            if event.event_type == "W_rec":
                # print("%.2f: worker %d received %d-th data"
                #       % (self.currentTime, event.data["index"], event.data["k_iter"]))
                worker = self.worker_list[event.data["index"]]
                worker.xdata["coordinates"] = event.data["coordinates"].copy()
                new_event = Event(self.currentTime + worker.generateCompleteTime(), event_type="W_cpt",
                                  data=event.data.copy())
                self.schedule.enqueue(new_event)
            elif event.event_type == "W_cpt":
                # print("%.2f: worker %d processed %d-th data"
                #       % (self.currentTime, event.data["index"], event.data["k_iter"]))
                worker = self.worker_list[event.data["index"]]
                worker.update()
                event.data["gradient"] = worker.gdata["gradient"].copy()
                new_event = Event(self.currentTime + worker.generateSendTime(), event_type="B_rec",
                                  data=event.data.copy())
                self.schedule.enqueue(new_event)
            elif event.event_type == "B_rec":
                # print("%.2f: Buffer received form worker %d based on %d-th data"
                #       % (self.currentTime, event.data["index"], event.data["k_iter"]))
                self.master.buffer.receive(event.data)
                if self.master.is_idle and len(self.master.buffer.datalist)>=self.master.bufferthresh:
                    self.master.fetch_data()
                    new_event = Event(self.master.generateCompleteTime() + self.currentTime, event_type="M_cpt",
                                      data={"index": -1, "k_iter": self.master.k_iter})
                    self.schedule.enqueue(new_event)
                    self.master.is_idle = False
            elif event.event_type == "M_cpt":

                self.master.update()
                self.master.is_idle = True
                self.master.k_iter += 1
                # print("%.2f: Master complete the %d-th data"
                #       % (self.currentTime, self.master.getKiter()))
                self.masterResultList.append(self.master.updatedx["coordinates"].copy())
                for i in self.master.sendlist:
                    worker = self.worker_list[i]
                    new_event = Event(worker.generateReceiveTime() + self.currentTime, event_type="W_rec",
                                      data={"index": i, "k_iter": self.master.getKiter(),
                                            "coordinates": self.master.updatedx["coordinates"].copy()})
                    self.schedule.enqueue(new_event)
                    # print("%.2f: Send %d-th data to worker %d" % (self.currentTime, self.master.k_iter, worker.index))
                self.master.sendlist = []
                if self.master.buffer.is_empty():
                    pass
                else:
                    self.master.fetch_data()
                    new_event = Event(self.master.generateCompleteTime() + self.currentTime, event_type="M_cpt",
                                      data={"index": -1, "k_iter": self.master.getKiter()})
                    self.schedule.enqueue(new_event)
                    self.master.is_idle = False

            else:
                raise TypeError
        except TypeError:
            print("Something goes wrong, event is invalid!!!")


def least_square_test():
    np.random.seed(100)
    numWorker = 5
    msize=40
    nsize=10
    bt = 3
    eachsize = msize//numWorker
    A = np.random.randn(msize, nsize)
    x = np.random.randn(nsize)
    b = A.dot(x)
    Aparts = [A[i*eachsize:(i+1)*eachsize] for i in range(numWorker)]
    bparts = [b[i*eachsize:(i+1)*eachsize] for i in range(numWorker)]
    workerlist = []
    for i in range(numWorker):
        def function(i):
            def operator(x):
                return Aparts[i].T.dot(Aparts[i].dot(x) - bparts[i])
            return operator
        workerlist.append(Worker(i, rcvDelay=0.1, procDelay=0.1, sndDelay=0.1,
                         func=function(i)))

    def init(self):
        self.updatedx = {"coordinates": np.zeros(nsize)}
        self.xdata = {"gradients": [np.zeros(nsize) for _ in range(numWorker)],
                      "old_coordinates": np.zeros(nsize)}

    def grad_update(self):
        sum_gradient = 0
        for gradient in self.xdata["gradients"]:
            sum_gradient += gradient
        tmp = self.updatedx["coordinates"]-0.001 * sum_gradient \
                +0.5*(self.updatedx["coordinates"]-self.xdata["old_coordinates"])
        self.xdata["old_coordinates"] = self.updatedx["coordinates"].copy()
        self.updatedx["coordinates"] = tmp.copy()


    master = Master(procDelay=0.1, initx=init, updatex=grad_update, bufferthresh=bt)
    sd = ScheduleDriver(workerlist, master)

    for i in range(10000):
        sd.next_event()
    plt.semilogy(np.sqrt(((np.array(sd.masterResultList) - x) ** 2).sum(1)) / np.linalg.norm(x))
    plt.show()


def lasso_test(stepsize=0.001,lambda1=0.1):
    np.random.seed(100)
    numWorker = 1
    msize = 100
    nsize = 300
    bt = 0
    eachsize = msize // numWorker
    A = np.random.randn(msize, nsize)
    x = np.zeros(nsize)
    x[np.random.permutation(nsize)[:10]] = 1
    b = A.dot(x)
    Aparts = [A[i * eachsize:(i + 1) * eachsize] for i in range(numWorker)]
    bparts = [b[i * eachsize:(i + 1) * eachsize] for i in range(numWorker)]
    workerlist = []
    for i in range(numWorker):
        def function(i):
            def operator(x):
                return Aparts[i].T.dot(Aparts[i].dot(x) - bparts[i])
            return operator

        workerlist.append(Worker(i, rcvDelay=0.1, procDelay=0.1, sndDelay=0.1,
                                 func=function(i)))

    def init(self):
        self.updatedx = {"coordinates": np.ones(nsize)}
        self.xdata = {"gradients": [np.zeros(nsize) for _ in range(numWorker)],
                      "xold": np.ones(nsize),"xnew":np.ones(nsize)}

    def prox_l1(y, alpha):
        ## \arg\min\|x\|_1+\frac{1}{2\alpha}\|x-y\|_2^2
        x = y.copy()
        x[y > alpha] = x[y > alpha] - alpha
        x[y < -alpha] = x[y < -alpha] + alpha
        x[abs(y) - alpha <= 0] = 0
        return x

    def grad_update(self):
        sum_gradient = 0
        eta = 0.0
        for gradient in self.xdata["gradients"]:
            sum_gradient += gradient

        self.xdata["xold"] = self.xdata["xnew"].copy()
        self.xdata["xnew"] = prox_l1(self.updatedx["coordinates"] - stepsize * sum_gradient,stepsize*lambda1)

        #self.xdata["old_coordinates"] = self.updatedx["coordinates"].copy()
        self.updatedx["coordinates"] = self.xdata["xnew"] + eta *(self.xdata["xnew"]-self.xdata["xold"])

    master = Master(procDelay=0.1, initx=init, updatex=grad_update, bufferthresh=bt)
    sd = ScheduleDriver(workerlist, master)

    for i in range(100000):
        sd.next_event()
    
    return np.sqrt(((np.array(sd.masterResultList) - x) ** 2).sum(1)) / np.linalg.norm(x)


def lasso_double_test(stepsize=0.001,eta1=0.1,eta2=0.1,lambda1=0.1):
    np.random.seed(100)
    numWorker = 4
    msize = 300
    nsize = 1000
    bt = 0.001
    eachsize = msize // numWorker
    A = np.random.randn(msize, nsize)
    x = np.zeros(nsize)
    x[np.random.permutation(nsize)[:10]] = 1
    b = A.dot(x)
    Aparts = [A[i * eachsize:(i + 1) * eachsize] for i in range(numWorker)]
    bparts = [b[i * eachsize:(i + 1) * eachsize] for i in range(numWorker)]

    def F(x):
        return 0.5*np.linalg.norm(A.dot(np.array(x).reshape(1000,-1)) - np.array(b).reshape(300,-1),ord=None, axis=None)**2 + lambda1 * np.linalg.norm(x , ord=1, axis=None)
    # print('The original F is:',  F(x))
    
    workerlist = []
    for i in range(numWorker):
        def function(i):
            def operator(x):
                return Aparts[i].T.dot(Aparts[i].dot(x) - bparts[i])
            return operator

        workerlist.append(Worker(i, rcvDelay=0.001, procDelay=0.001, sndDelay=0.001,
                                 func=function(i)))
    def init(self):
        self.updatedx = {"coordinates": np.ones(nsize)}
        self.xdata = {"gradients": [np.zeros(nsize) for _ in range(numWorker)],
                      "xold": np.ones(nsize),"xnew":np.ones(nsize),
                      "zold": np.ones(nsize),"znew":np.ones(nsize),
                      "y": np.ones(nsize)}

    def prox_l1(y, alpha):
        ## \arg\min\|x\|_1+\frac{1}{2\alpha}\|x-y\|_2^2
        x = y.copy()
        x[y > alpha] = x[y > alpha] - alpha
        x[y < -alpha] = x[y < -alpha] + alpha
        x[abs(y) - alpha <= 0] = 0
        return x

    def grad_update(self):
        sum_gradient = 0
        for gradient in self.xdata["gradients"]:
            sum_gradient += gradient
        self.xdata["y"] = self.xdata["xnew"]+eta1*(self.xdata["xnew"]-self.xdata["xold"])
        self.xdata["zold"] = self.xdata["znew"].copy()
        self.xdata["znew"] = prox_l1( self.xdata["y"] - stepsize * sum_gradient,stepsize*lambda1)
        self.xdata["xold"] = self.xdata["xnew"].copy()
        self.xdata["xnew"] = self.xdata["znew"]+eta2*(self.xdata["znew"]-self.xdata["zold"])
        self.updatedx["coordinates"] = self.xdata["xnew"].copy()

    master = Master(procDelay=0.001, initx=init, updatex=grad_update, bufferthresh=bt)
    sd = ScheduleDriver(workerlist, master)
    iterates = 100000
    loss_old = 1e15
    F_loss = []
    for i in range(iterates):
        sd.next_event()
        print('The '+ str(i) +'-th F is:', F(np.array(sd.master.updatedx["coordinates"])))

    for x in sd.masterResultList:
        loss = F(np.array(x))
        print('The F is:', loss)
        # if loss_old < loss:
        #     break
        # loss_old = loss
        F_loss.append(loss)

    return np.sqrt(((np.array(sd.masterResultList) - x) ** 2).sum(1)) / np.linalg.norm(x), np.array(F_loss)



def FISTAwithBacktracking(stepsize=1, lambda1=0.1, restart=True):
    np.random.seed(100)
    msize = 300
    nsize = 1000
    A = np.random.randn(msize, nsize)
    x = np.zeros(nsize)
    x[np.random.permutation(nsize)[:10]] = 1
    b = A.dot(x)

    def prox_l1(y, alpha):
     ## \arg\min\|x\|_1+\frac{1}{2\alpha}\|x-y\|_2^2
        x = y.copy()
        x[y > alpha] = x[y > alpha] - alpha
        x[y < -alpha] = x[y < -alpha] + alpha
        x[abs(y) - alpha <= 0] = 0
        return x


    def F(x):
        return 0.5*np.linalg.norm(A.dot(np.array(x).reshape(1000,-1)) - np.array(b).reshape(300,-1),ord=None, axis=None)**2 + lambda1 * np.linalg.norm(x , ord=1, axis=None)
    print('The original F is:',  F(x))

    def Q(u,v,L0):
        A_value = 0.5*np.linalg.norm(A.dot(np.array(v).reshape(1000,-1)) - np.array(b).reshape(300,-1),ord=None, axis=None)**2
        B = (u-v).T.dot(A.T.dot(A.dot(v) - b))
        C = 0.5*L0*np.linalg.norm(u-v,ord=None, axis=None)**2
        D =lambda1 * np.linalg.norm(u , ord=1, axis=None)
        return np.array(A_value + B + C + D)
    
    iterations = 10000
    eta = 1
    xdata = {"xold": np.ones(nsize),"xnew": np.ones(nsize),"yold": np.ones(nsize),"ynew": np.ones(nsize)}
    masterResultList = []
    F_loss_list = np.zeros((iterations,1))
    loss_old = 1e4
    for i in range(iterations):
        stepsize_new = stepsize
        while restart == True:
            xdata["gradients"] = A.T.dot(A.dot(xdata["yold"]) - b)
            zk = prox_l1(xdata["yold"]- stepsize_new * xdata["gradients"], stepsize_new*lambda1)
            # print('Q', Q(zk,xdata["yold"], stepsize))
            # print('F', F(zk))
            # print('Q', Q(zk, xdata["yold"], 1/stepsize_new))
            if F(zk) <= Q(zk, xdata["yold"], 1/stepsize_new):
                break
            stepsize_new = stepsize_new*0.5
            stepsize = stepsize_new
        
        xdata["gradients"] = A.T.dot(A.dot(xdata["yold"]) - b)
        xdata["xnew"] = prox_l1(xdata["yold"]- stepsize * xdata["gradients"], lambda1 * stepsize)
        eta_new = 0.5*(1 + np.sqrt(1 + 4 * eta**2))
        xdata["ynew"] = xdata["xnew"] + (eta -1)/eta_new * (xdata["xnew"] - xdata["xold"])
        masterResultList.append(xdata["ynew"].copy())
        xdata["yold"] = xdata["ynew"].copy()
        xdata["xold"] = xdata["xnew"].copy()
        eta = eta_new.copy()
        
        F_loss = F(xdata["ynew"])
        print('The '+str(i)+'-th F is:', F_loss)
        F_loss_list[i,:] = F_loss
        # if loss_old < F_loss:
        #     break
        # loss_old = F_loss
    return np.array(F_loss_list)

def testing():
    print("Testing worker:")
    w = Worker(rcvDelay=1, procDelay=10, sndDelay=100)
    print(w.generateCompleteTime())
    print(w.generateReceiveTime())
    print(w.generateSendTime())
    print("-" * 20)

    print("Testing Master")
    m = Master(procDelay=10)
    print(m.generateCompleteTime())
    print("-" * 20)

    print("Testing Schedule")
    schedule = Schedule()
    schedule.enqueue(Event(2.3, "W_rec"))
    schedule.enqueue(Event(5.3, "W_rec"))
    schedule.enqueue(Event(1.3, "W_rec"))
    schedule.enqueue(Event(2.5, "W_rec"))
    schedule.enqueue(Event(3.3, "W_rec"))
    schedule.display()
    schedule.dequeue().display()
    schedule.dequeue().display()
    schedule.dequeue().display()
    schedule.dequeue().display()
    schedule.dequeue().display()
    schedule.dequeue()
    print("-" * 20)
    print("Testing Schedule_driver")

    workerlist = [Worker(0, rcvDelay=10, procDelay=1, sndDelay=10),
                  Worker(1, rcvDelay=15, procDelay=2, sndDelay=15),
                  Worker(2, rcvDelay=20, procDelay=3, sndDelay=20)]
    master = Master(procDelay=30)
    sd = ScheduleDriver(workerlist, master)

    for i in range(50):
        sd.next_event()

if __name__ == "__main__":
    
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    # testing()
    # least_square_test()
    #plt.axis([-100,250000,0.0005,50])
    stepsize_setting = 0.0001
    lambda_setting = 2

    F_loss_list0 = FISTAwithBacktracking(stepsize=stepsize_setting, lambda1=lambda_setting, restart=False)
    F_loss_list = FISTAwithBacktracking(stepsize=1, lambda1=lambda_setting,restart=True)
    r0, r0_Loss = lasso_double_test(stepsize=stepsize_setting, eta1=0.0, eta2=0.0, lambda1=lambda_setting)
    r1, r1_Loss = lasso_double_test(stepsize=stepsize_setting, eta1=0.1, eta2=0.0, lambda1=lambda_setting)
    r2, r2_Loss = lasso_double_test(stepsize=stepsize_setting, eta1=0.2, eta2=0.0, lambda1=lambda_setting)
    r3, r3_Loss = lasso_double_test(stepsize=stepsize_setting, eta1=0.3, eta2=0.0, lambda1=lambda_setting)
    r4, r4_Loss = lasso_double_test(stepsize=stepsize_setting, eta1=0.4, eta2=0.0, lambda1=lambda_setting)
    
    np.save(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_F_loss_list_0.npy', F_loss_list0)
    np.save(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_F_loss_list.npy', F_loss_list)
    np.savez(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_x.npz', r0, r1, r2, r3, r4)
    np.savez(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_Loss.npz', r0_Loss, r1_Loss, r2_Loss, r3_Loss, r4_Loss)
    
    F_loss_list0 = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+ str(stepsize_setting)+'_F_loss_list_0.npy')
    F_loss_list = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_F_loss_list.npy')
    r0 = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_x.npz')['arr_0']
    r1 = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_x.npz')['arr_1']
    r2 = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_x.npz')['arr_2']
    r3 = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_x.npz')['arr_3']
    r4 = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_x.npz')['arr_4']
    r0_Loss = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_Loss.npz')['arr_0']
    r1_Loss = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_Loss.npz')['arr_1']
    r2_Loss = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_Loss.npz')['arr_2']
    r3_Loss = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_Loss.npz')['arr_3']
    r4_Loss = np.load(r'data/lambda_'+ str(lambda_setting) +'_alpha_'+str(stepsize_setting)+'_Loss.npz')['arr_4']
    
    plt.figure(num=1)
    maxiter_x = 10000
    markernumber1 = 200
    plt.semilogy(r0[0:maxiter_x],  color='k', linestyle="--", marker='.', markevery=markernumber1, label=r'PIAG')
    plt.semilogy(r1[0:maxiter_x],  color='gold', linestyle="-", marker='x', markevery=markernumber1, label=r'iPIAG with $\beta=0.1$')
    plt.semilogy(r2[0:maxiter_x],  color='darkseagreen', linestyle="-", marker='s', markevery=markernumber1, label=r'iPIAG with $\beta=0.2$')
    plt.semilogy(r3[0:maxiter_x],  color='lightpink', linestyle="-", marker='*', markevery=markernumber1, label=r'iPIAG with $\beta=0.3$')
    plt.semilogy(r4[0:maxiter_x],  color='skyblue', linestyle="-", marker='o', markevery=markernumber1, label=r'iPIAG with $\beta=0.4$')
    plt.legend(loc='best')
    plt.ylabel( r"$\|x_k-x^\ast\|/\|x^\ast\|$",
               fontsize=16, color='black')
    plt.xlabel(r"Iterate $k$")
    plt.savefig('fig/lasso_x_lambda_2_alpha_0001.eps')
    
    plt.figure(num=2)
    plt.ylim(1e-14,1e6)
    F_star = F_loss_list[-1,:]
    # F_star = 0
    print('F_star', F_star)
    maxiter=10000
    markernumber2 = 200
    # print(np.where(r0_Loss<F_star, F_star, r0_Loss))
    # plt.semilogy(F_loss_list0[0:maxiter],  color='silver', linestyle="--", marker='^', markevery=1000, label=r'FISTA')
    # plt.semilogy(F_loss_list[0:maxiter],  color='r', linestyle="--", marker='d', markevery=1000, label=r'FISTA with backtracking')
    plt.semilogy(r0_Loss[0:maxiter]-F_star, color='k', linestyle="--", marker='.', markevery=markernumber2, label=r'PIAG')
    plt.semilogy(r1_Loss[0:maxiter]-F_star,  color='gold', linestyle="-", marker='x', markevery=markernumber2, label= r'iPIAG with $\beta=0.1$')
    plt.semilogy(r2_Loss[0:maxiter]-F_star,  color='darkseagreen', linestyle="-", marker='s', markevery=markernumber2, label= r'iPIAG with $\beta=0.2$')
    plt.semilogy(r3_Loss[0:maxiter]-F_star,  color='lightpink', linestyle="-", marker='*', markevery=markernumber2, label= r'iPIAG with $\beta=0.3$')
    plt.semilogy(r4_Loss[0:maxiter]-F_star,  color='skyblue', linestyle="-", marker='o', markevery=markernumber2, label= r'iPIAG with $\beta=0.4$')
    plt.legend(loc='best')
    # plt.ylabel( r"$F(x_k)$",fontsize=16, color='black')
    plt.ylabel(r"$F(x_k)-F(x^{\ast})$",fontsize=16, color='black')
    plt.xlabel(r"Iterate $k$")
    plt.savefig("fig/lasso_loss_lambda_2_alpha_0001.eps")