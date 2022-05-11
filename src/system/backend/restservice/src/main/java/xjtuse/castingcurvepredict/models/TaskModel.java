package xjtuse.castingcurvepredict.models;

import java.time.Instant;
import java.util.Date;
import java.util.Vector;

import javax.xml.crypto.Data;

import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;

public class TaskModel {

    private Vector<ITaskEventListener> mListeners = new Vector<>();
    private TaskStatus mStatus = TaskStatus.Created;
    private Date mStartTime = Date.from(Instant.now());
    private Date mStopTime;
    private long mId;
    private Double mLoss = Double.MAX_VALUE;
    private int mEpoch = 0;
    private long _modelId;
    ICastingGenerator _generator;

    public TaskModel(long id, double loss, int epoch, TaskStatus status, Date startTime, Date stopTime, long modelId) {
        mId = id;
        mLoss = loss;
        mEpoch = epoch;
        mStatus = status;
        mStartTime = startTime;
        mStopTime = stopTime;
        _modelId = modelId;
    }

    public TaskModel(long id, double loss, int epoch, TaskStatus status, Date startTime, long modelId) {
        mId = id;
        mLoss = loss;
        mEpoch = epoch;
        mStatus = status;
        mStartTime = startTime;
        _modelId = modelId;
    }

    public TaskModel(long taskId, long modelId) {
        mId = taskId;
        _modelId = modelId;
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

    public long getModelId() {
        return _modelId;
    }

    public void AddEventListener(ITaskEventListener listener) {
        mListeners.add(listener);
    }

    public void Start(ICastingGenerator generator) {
        if (generator != null) {
            _generator = generator;
            for (ITaskEventListener listener : mListeners) {
                listener.onTaskStarting(this);
            }

            setStartTime(Date.from(Instant.now()));
            generator.train(this);

            for (ITaskEventListener listener : mListeners) {
                listener.onTaskStarted(this);
            }
        }

    }

    public void Stop() {
        // TODO: stop the task.
        if (_generator != null) {
            _generator.stop(this);
        }

        setStopTime(Date.from(Instant.now()));

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
