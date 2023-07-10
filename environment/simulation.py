import simpy, os, random, copy
import pandas as pd
import numpy as np
from collections import Counter


class Job:
    def __init__(self, name, time, job_type=None, due_date=None):
        # 해당 job의 이름
        self.name = name
        # 해당 job의 작업시간
        self.processing_time = time
        self.job_type = job_type
        self.due_date = due_date
        self.completion_time = 0
        self.arrival_time = None
        self.past = 0
        self.sink_just = True




class Source:
    def __init__(self, name, env, routing, monitor, jt_dict, p_j, K, machine_num):
        self.env = env
        self.name = name
        self.routing = routing
        self.monitor = monitor
        self.total_p_j = p_j
        self.p_j = p_j[int(name[-1])]  # mean of exponential distribution
        self.jobs = jt_dict["JobType {0}".format(int(name[-1]))]
        self.machine_num = machine_num
        self.len_jobs = len(copy.deepcopy(self.jobs))
        self.iat_list = self._cal_iat()

        self.non_processed_job = copy.deepcopy(jt_dict["JobType {0}".format(int(name[-1]))])

        # set duedate
        start_time = 0
        self.due_date = list()
        for i in range(len(self.jobs)):
            start_time += self.iat_list[i]
            self.jobs[i].due_date = start_time + self.p_j * K
            self.due_date.append(start_time + self.p_j * K)

        self.env.process(self.generate())

    def generate(self):
        while len(self.jobs):
            job = self.jobs.pop(0)
            iat = self.iat_list.pop(0)
            yield self.env.timeout(iat)
            job.arrival_time = copy.deepcopy(self.env.now)
            job.past = copy.deepcopy(self.env.now)
            self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Created", job=job.name)
            self.routing.queue.put(job)
            self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Put in Routing Class", job=job.name,
                                queue=len(self.routing.queue.items))
            self.routing.created += 1
            self.env.process(self.routing.run(location="Source"))

    def _cal_iat(self):
        total_p_j = np.sum(self.total_p_j)
        lambda_u = 2 * self.machine_num / (total_p_j - 1)
        arrival_rate = list(np.random.uniform(1, lambda_u, size=self.len_jobs))
        iat_list = [1/ar for ar in arrival_rate]
        return iat_list


class Process:
    def __init__(self, env, name, sink, routing, monitor):
        self.env = env
        self.name = name
        self.sink = sink
        self.routing = routing
        self.monitor = monitor

        self.queue = simpy.Store(env)
        self.idle = True
        self.job = None
        self.planned_finish_time = 0

        env.process(self.run())

    def run(self):
        while True:
            self.job = yield self.queue.get()
            self.idle = False
            self.monitor.record(time=self.env.now, jobtype=self.job.job_type, event="Work Start", job=self.job.name,
                                machine=self.name)
            processing_time = self.job.processing_time[int(self.name[-1])]
            self.planned_finish_time = self.env.now + processing_time
            yield self.env.timeout(processing_time)
            self.job.completion_time = self.env.now
            self.monitor.record(time=self.env.now, jobtype=self.job.job_type, event="Work Finish", job=self.job.name,
                                machine=self.name)

            self.sink.put(self.job)
            self.idle = True
            self.job = None

            if len(self.routing.queue.items) > 0:
                self.monitor.record(time=self.env.now, event="Request Routing for Job", machine=self.name,
                                    queue=len(self.routing.queue.items))
                yield self.env.process(self.routing.run(location=self.name))
            elif (len(self.queue.items) == 0) and (self.routing.created == self.sink.end_num):
                break


class Routing:
    def __init__(self, env, process_dict, source_dict, monitor, weight):
        self.env = env
        self.process_dict = process_dict
        self.source_dict = source_dict
        self.monitor = monitor
        self.weight = weight

        self.created = 0

        self.queue = simpy.FilterStore(env)
        self.waiting = env.event()

        self.indicator = False
        self.decision = False

        self.idle = False
        self.job = None

        #self.mapping = {0: "WSPT", 1: "WMDD", 2: "ATC", 3: "WCOVERT"}

    def run(self, location="Source"):
        machine_idle = [machine.idle for machine in self.process_dict.values()]
        if location == "Source":  # job -> machine 선택
            if any(machine_idle):
                job = yield self.queue.get()

                self.indicator = True
                self.decision = self.env.event()
                routing_rule = yield self.decision
                self.decision = None
                self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Routing Start", job=job.name,
                                    memo="{0},  machine 선택".format(routing_rule))
                # routing_rule_number = np.random.randint(low=0, high=4)
                # routing_rule = self.mapping[routing_rule_number]

                next_machine = None
                if self.action_mode == 'heuristic':
                    if routing_rule == "WSPT":
                        next_machine = yield self.env.process(self.WSPT(location=location, idle=machine_idle, job=job))
                    elif routing_rule == "WMDD":
                        next_machine = yield self.env.process(self.WMDD(location=location, idle=machine_idle, job=job))
                    elif routing_rule == "ATC":
                        next_machine = yield self.env.process(self.ATC(location=location, idle=machine_idle, job=job))
                    elif routing_rule == "WCOVERT":
                        next_machine = yield self.env.process(self.WCOVERT(location=location, idle=machine_idle, job=job))
                    self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Routing Finish", job=job.name,
                                        machine="Machine {0}".format(next_machine))
                    self.process_dict["Machine {0}".format(next_machine)].queue.put(job)

                elif self.action_mode == 'WCOVERT':
                    next_machine = yield self.env.process(self.WCOVERT(location=location, idle=machine_idle, job=job,
                                                                       k_t = routing_rule))
                    self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Routing Finish", job=job.name,
                                        machine="Machine {0}".format(next_machine))
                    self.process_dict["Machine {0}".format(next_machine)].queue.put(job)

                # else:


        else:  # machine -> job 선택
            if len(self.queue.items) > 0:
                self.indicator = True
                # routing_rule_number = np.random.randint(low=0, high=4)
                # routing_rule = self.mapping[routing_rule_number]
                self.decision = self.env.event()
                routing_rule = yield self.decision
                self.decision = None

                self.monitor.record(time=self.env.now, event="Routing Start", machine=location, memo="{0} Job 선택".format(routing_rule))
                next_job = None
                if self.action_mode == 'heuristic':
                    if routing_rule == "WSPT":
                        next_job = yield self.env.process(self.WSPT(location=location))
                    elif routing_rule == "WMDD":
                        next_job = yield self.env.process(self.WMDD(location=location))
                    elif routing_rule == "ATC":
                          next_job = yield self.env.process(self.ATC(location=location, idle=machine_idle))
                    elif routing_rule == "WCOVERT":
                        next_job = yield self.env.process(self.WCOVERT(location=location))
                if self.action_mode == 'WCOVERT':
                    next_job = yield self.env.process(self.WCOVERT(location=location, k_t = routing_rule))

                self.monitor.record(time=self.env.now, jobtype=next_job.job_type, event="Routing Finish",
                                    job=next_job.name, machine=location)

                self.process_dict[location].queue.put(next_job)

    def WSPT(self, location="Source", idle=None, job=None):
        if location == "Source":  # job -> machine 선택 => output : machine index
            min_processing_time = 1e10
            min_machine_idx = None
            jt = job.job_type

            for idx in range(len(idle)):
                if idle[idx]:
                    wpt = job.processing_time[idx] / self.weight[jt]
                    if wpt < min_processing_time:
                        min_processing_time = wpt
                        min_machine_idx = idx

            return min_machine_idx

        else:  # machine -> job 선택 => output : job
            job_list = list(copy.deepcopy(self.queue.items))
            min_processing_time = 1e10
            min_job_name = None

            for job_q in job_list:
                jt = job_q.job_type
                wpt = job_q.processing_time[int(location[-1])] / self.weight[jt]

                if wpt < min_processing_time:
                    min_processing_time = wpt
                    min_job_name = job_q.name

            next_job = yield self.queue.get(lambda x: x.name == min_job_name)

            return next_job

    def WMDD(self, location="Source", idle=None, job=None):
        if location == "Source":  # job -> machine 선택 => output : machine index
            min_wdd = 1e10
            min_machine_idx = None
            jt = job.job_type

            for idx in range(len(idle)):
                if idle[idx]:
                    wdd = max(job.processing_time[idx], job.due_date - self.env.now) / self.weight[jt]
                    if wdd < min_wdd:
                        min_wdd = wdd
                        min_machine_idx = idx

            return min_machine_idx

        else:  # machine -> job 선택 => output : job
            job_list = list(copy.deepcopy(self.queue.items))
            min_wdd = 1e10
            min_job_name = None

            for job_q in job_list:
                jt = job_q.job_type
                wdd = max(job_q.processing_time[int(location[-1])], job_q.due_date - self.env.now) / self.weight[jt]

                if wdd < min_wdd:
                    min_wdd = wdd
                    min_job_name = job_q.name

            next_job = yield self.queue.get(lambda x: x.name == min_job_name)

            return next_job

    def ATC(self, location="Source", idle=None, job=None, h = 1):
        p_temp = []
        if self.queue.items is not None:
            for wj in list(self.queue.items):
                p_temp.append(np.average(wj.processing_time))
        p = 0 if p_temp is None or len(p_temp) == 0 else np.average(p_temp)  # p 충분히 크면 ATC==WSPT
        if location == "Source":  # job -> machine 선택 => output : machine index
            max_wa = -1
            max_machine_idx = None
            jt = job.job_type
            non_processed_job = self.source_dict["Source {0}".format(jt)].jobs # waiting jobs of JT
            temp = list()
            for idx in range(len(idle)):
                if idle[idx]:
                    p_ij = job.processing_time[idx]
                    if len(non_processed_job) > 0: # X
                        # p = np.average([non_job.processing_time[idx] for non_job in non_processed_job]) #??????
                        wa = self.weight[jt] / p_ij * np.exp(-(max(job.due_date - p_ij - self.env.now, 0) / (h * p))) \
                            if p > 0 else 0
                        if wa > max_wa:
                            max_wa = wa
                            max_machine_idx = idx
                    else: # X
                        print("else: len(non_processed_job) <= 0")
                        temp.append(idx)


            if len(temp) > 0: # X
                print("error")
                max_machine_idx = random.choice(temp)

            return max_machine_idx

        else:  # machine(location) -> job 선택 => output : job # idle is None
            job_list = list(copy.deepcopy(self.queue.items))
            max_wa = -1
            max_job_name = None

            for job_q in job_list:
                jt = job_q.job_type

                # p = np.average([non_job.processing_time[int(location[-1])] for non_job in non_processed_job])
                wa = self.weight[jt] / job_q.processing_time[int(location[-1])] * np.exp(
                    -(max(job_q.due_date - job_q.processing_time[int(location[-1])] - self.env.now, 0) / (h * p))) \
                    if p > 0 else 0

                if wa > max_wa:
                    max_wa = wa
                    max_job_name = job_q.name
            # if max_job_name is None:
            #     print("errorrrrrrrrrrrrrrrr")
            #     max_job_name = random.choice(temp)
            next_job = yield self.queue.get(lambda x: x.name == max_job_name)
            return next_job

    def WCOVERT(self, location="Source", idle=None, job=None, k_t = 14.5):
        if location == "Source":  # job -> machine 선택 => output : machine index
            max_wt = -1
            max_machine_idx = None
            jt = job.job_type
            for idx in range(len(idle)):
                if idle[idx]:
                    p_ij = job.processing_time[idx]
                    temp_wt = max(job.due_date - p_ij - self.env.now, 0) / (k_t * p_ij)
                    temp_wt = max(1 - temp_wt, 0)
                    wt = self.weight[jt] / p_ij * temp_wt

                    if wt > max_wt:
                        max_wt = wt
                        max_machine_idx = idx

            return max_machine_idx

        else:  # machine -> job 선택 => output : job
            job_list = list(copy.deepcopy(self.queue.items))
            max_wt = -1
            max_job_name = None

            for job_q in job_list:
                jt = job_q.job_type
                p_ij = job_q.processing_time[int(location[-1])]
                wt = self.weight[jt] * np.exp(1 - (max(job_q.due_date - p_ij - self.env.now, 0) / (k_t * p_ij))) / p_ij

                if wt > max_wt:
                    max_wt = wt
                    max_job_name = job_q.name

            next_job = yield self.queue.get(lambda x: x.name == max_job_name)

            return next_job


class Sink:
    def __init__(self, env, monitor, jt_dict, end_num, source_dict, weight):
        self.env = env
        self.monitor = monitor
        self.jt_dict = jt_dict
        self.idle = False
        self.end_num = end_num
        self.source_dict = source_dict
        self.weight = weight

        # JobType 별 작업이 종료된 Job의 수
        self.finished = {jt: 0 for jt in self.jt_dict.keys()}
        self.finished_job = 0

        self.job_list = list()
        ######### for state feature 9 #####################
        self.tardiness_jt = {}
        self.tardiness_jt_cnt = {}
        ###################################################

    def put(self, job):
        self.finished["JobType {0}".format(job.job_type)] += 1  # jobtype 별 종료 개수

        self.source_dict["Source {0}".format(job.job_type)].non_processed_job = [non_processed for non_processed in
                                                                                 self.source_dict["Source {0}".format(
                                                                                     job.job_type)].non_processed_job if
                                                                                 non_processed.name != job.name]

        self.finished_job += 1  # 전체 종료 개수
        self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Completed", job=job.name,
                            memo=max(self.env.now - job.due_date, 0))
        self.job_list.append(job)

        self.monitor.tardiness += self.weight[job.job_type] * min(0, job.due_date - self.env.now)
        ####################### for state feature 9 ################################
        # if job.job_type not in list(self.tardiness_jt.keys()):
        #     self.tardiness_jt[job.job_type] = min(0, job.due_date - self.env.now)
        #     self.tardiness_jt_cnt[job.job_type] = 1
        # else:
        #     self.tardiness_jt[job.job_type] += min(0, job.due_date - self.env.now)
        #     self.tardiness_jt_cnt[job.job_type] += 1
        #############################################################################

class Monitor:
    def __init__(self, filepath):
        self.time = list()
        self.jobtype = list()
        self.event = list()
        self.job = list()
        self.machine = list()
        self.queue = list()
        self.memo = list()

        self.filepath = filepath

        self.tardiness = None

    def record(self, time=None, jobtype=None, event=None, job=None, machine=None, queue=None, memo=None):
        self.time.append(round(time, 2))
        self.jobtype.append(jobtype)
        self.event.append(event)
        self.job.append(job)
        self.machine.append(machine)
        self.queue.append(queue)
        self.memo.append(memo)

    def save_tracer(self):
        event_tracer = pd.DataFrame(columns=["Time", "JobType", "Event", "Job", "Machine", "Queue", "Memo"])
        event_tracer["Time"] = self.time
        event_tracer["JobType"] = self.jobtype
        event_tracer["Event"] = self.event
        event_tracer["Job"] = self.job
        event_tracer["Machine"] = self.machine
        event_tracer["Queue"] = self.queue
        event_tracer["Memo"] = self.memo
        event_tracer.to_csv(self.filepath, encoding='utf-8-sig')

    def reset(self):
        self.time = list()
        self.jobtype = list()
        self.event = list()
        self.job = list()
        self.machine = list()
        self.queue = list()
        self.memo = list()

        self.tardiness = 0

