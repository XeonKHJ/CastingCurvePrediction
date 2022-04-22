package xjtuse.castingcurvepredict.castingpredictiors.dummyimpl;

import java.util.Date;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.Semaphore;

import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;

// 以流的方式实现状态管理。
// 一旦状态被读取后，就会从缓冲区中删除，不可再读。
// 线程安全。
public class StreamStatusManager implements IStatusManager {

    SortedMap<Date, Double> mLosses = new TreeMap<Date, Double>();
    SortedMap<Date, Integer> mEpochs = new TreeMap<Date, Integer>();
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
        return mLosses;
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

}
