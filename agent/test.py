


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

def WCOVERT(self, location="Source", idle=None, job=None, k_t = 14.5):

    if location == "Source":  # job -> machine 선택 => output : machine index
        max_wt = -1
        max_machine_idx = None
        jt = job.job_type
        for idx in range(len(idle)):
            if idle[idx]:
                p_ij = job.processing_time[idx]
                temp_wt = max(job.due_date - p_ij - self.env.now, 0)
                temp_wt = temp_wt / (k_t * p_ij)
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
            wt = self.weight[jt] / p_ij * np.exp(1 - (max(job_q.due_date - p_ij - self.env.now, 0) / (k_t * p_ij)))

            if wt > max_wt:
                max_wt = wt
                max_job_name = job_q.name

        next_job = yield self.queue.get(lambda x: x.name == max_job_name)

        return next_job
