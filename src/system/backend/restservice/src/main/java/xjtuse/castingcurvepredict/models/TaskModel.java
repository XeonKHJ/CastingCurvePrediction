package xjtuse.castingcurvepredict.models;

import java.util.Date;
import java.util.Vector;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;

public class TaskModel {

    private Vector<ITaskEventListener> mListeners = new Vector<>();
    private TaskStatus mStatus;
    private Date mStartTime;
    private Date mStopTime;
    private long mId;
    private Double mLoss;
    private int mEpoch;
    ICastingGenerator _generator;
    public TaskModel(long id, double loss, int epoch, TaskStatus status, Date startTime, Date stopTime) {
        mId = id;
        mLoss = loss;
        mEpoch = epoch;
        mStatus = status;
        mStartTime = startTime;
        mStopTime = stopTime;
    }

    public TaskModel(long id, double loss, int epoch, TaskStatus status, Date startTime) {
        mId = id;
        mLoss = loss;
        mEpoch = epoch;
        mStatus = status;
        mStartTime = startTime;
    }

    public TaskModel(long id) {
        mId = id;
    }

    // getter
    public long getId() {
        return mId;
    }

    public void setLoss(double value) {
        mLoss = value;
    }

    public double getLoss() {
        return mLoss;
    }

    public void setEpoch(int value) {
        mEpoch = value;
    }

    public int getEpoch() {
        return mEpoch;
    }

    public void setStatus(TaskStatus status) {
        mStatus = status;
    }

    public TaskStatus getStatus() {
        return mStatus;
    }

    public void setStartTime(Date value) {
        mStartTime = value;
    }

    public Date getStartTime() {
        return mStartTime;
    }

    public void setStopTime(Date value) {
        mStopTime = value;
    }

    public Date getStopTime() {
        return mStopTime;
    }

    public void AddEventListener(ITaskEventListener listener) {
        mListeners.add(listener);
    }

    public void Start(ICastingGenerator generator) {
        for (ITaskEventListener listener : mListeners) {
            listener.onTaskStarting(this);
        }
        
        generator.train(this);

        for (ITaskEventListener listener : mListeners) {
            listener.onTaskStarted(this);
        }
    }

    public void Stop() {
        // TODO: stop the task.
        if(_generator != null)
        {
            _generator.stop(this);
        }


        for (ITaskEventListener listener : mListeners) {
            listener.onTaskStopped(this);
        }
    }

    public void Pause() {
        // TODO 暂停任务
    }

    public void Update() {
        // TODO: 更新模型
    }
}
