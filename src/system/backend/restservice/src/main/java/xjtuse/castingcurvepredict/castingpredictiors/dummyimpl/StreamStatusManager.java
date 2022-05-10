package xjtuse.castingcurvepredict.castingpredictiors.dummyimpl;

import java.util.Date;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.Vector;
import java.util.concurrent.Semaphore;

import org.springframework.scheduling.config.Task;

import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManagerEventListener;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;
import xjtuse.castingcurvepredict.models.TaskModel;

// 以流的方式实现状态管理。
// 一旦状态被读取后，就会从缓冲区中删除，不可再读。
// 线程安全。
public class StreamStatusManager implements IStatusManager {

    SortedMap<Date, Double> mLosses = new TreeMap<Date, Double>();
    SortedMap<Date, Integer> mEpochs = new TreeMap<Date, Integer>();
    private TaskModel mTask;
    private Vector<IStatusManagerEventListener> mListeners = new Vector<>();
    TaskStatus mStatus;
    Semaphore mutex;

    @Override
    public void saveLosses(Map<Date, Double> losses) {
        // TODO 上锁
        for (Map.Entry<Date, Double> entry : losses.entrySet()) {
            Date mapKey = entry.getKey();
            Double mapValue = entry.getValue();
            mLosses.put(mapKey, mapValue);
        }
        // TODO 解锁
    }

    @Override
    public void saveStatus(TaskStatus status) {
        mStatus = status;
        mTask.setStatus(status);
        switch (status) {
            case Stopping:
                for (var listener : mListeners) {
                    listener.onTaskStopped(this);
                }
                break;
            default:
                break;
        }
    }

    @Override
    public void saveStatus(TaskStatus status, Date time) {
        mStatus = status;
    }

    @Override
    public void saveEpochs(Map<Date, Integer> epochs) {
        // TODO 上写锁
        for (Map.Entry<Date, Integer> entry : epochs.entrySet()) {
            Date mapKey = entry.getKey();
            Integer mapValue = entry.getValue();
            mEpochs.put(mapKey, mapValue);
        }
        // TODO 解写锁
    }

    @Override
    public SortedMap<Date, Double> readLosses() {
        SortedMap<Date, Double> result = new TreeMap<>();

        // 上锁
        for (var entry : mLosses.entrySet()) {
            Date mapKey = entry.getKey();
            double mapValue = entry.getValue();
            result.put(mapKey, mapValue);
        }
        mLosses.clear();

        return result;
    }

    @Override
    public SortedMap<Date, Integer> readEpochs() {
        // 做一次深拷贝
        SortedMap<Date, Integer> result = new TreeMap<Date, Integer>();

        // TODO 上锁
        for (Map.Entry<Date, Integer> entry : mEpochs.entrySet()) {
            Date mapKey = entry.getKey();
            Integer mapValue = entry.getValue();
            result.put(mapKey, mapValue);
        }
        mEpochs.clear();
        // TODO 解锁

        return result;
    }

    @Override
    public TaskStatus readStatus() {
        return mStatus;
    }

    @Override
    public void AddEventListener(IStatusManagerEventListener listener) {
        mListeners.add(listener);
    }

    @Override
    public void RemoveEventListenr(IStatusManagerEventListener listener) {
        mListeners.remove(listener);
    }

    @Override
    public void setTask(TaskModel task) {
        mTask = task;
    }

    @Override
    public TaskModel getTask() {
        return mTask;
    }
}
