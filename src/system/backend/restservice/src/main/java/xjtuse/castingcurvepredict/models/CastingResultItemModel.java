package xjtuse.castingcurvepredict.models;

import org.apache.tomcat.util.threads.StopPooledThreadException;

public class CastingResultItemModel {
    private String _datetime;

    // 冷凝管中钢液高度。
    private double _liqLevel;

    // 大包重量
    private double _ladleWeight;

    // 中包重量
    private double _tudishWeight;

    // 塞棒位置
    private double _stopperPos;

    public CastingResultItemModel(String datetime, double stopperPos, double liqLevel, double tudishWeight, double ladleWeight) {
        _datetime = datetime;
        _liqLevel = liqLevel;
        _stopperPos = stopperPos;
        _ladleWeight = ladleWeight;
        _tudishWeight = tudishWeight;
    }

    public CastingResultItemModel(String datetime, double stopperPos) {
        _datetime = datetime;
        _stopperPos = stopperPos;
    }

    public String getDatetime() {
        return _datetime;
    }

    public double getStopperPos() {
        return _stopperPos;
    }

    public double getLiqLevel() {
        return _liqLevel;
    }

    public double getTudishWeight()
    {
        return _tudishWeight;
    }

    public double getLadleWeight()
    {
        return _ladleWeight;
    }
}
