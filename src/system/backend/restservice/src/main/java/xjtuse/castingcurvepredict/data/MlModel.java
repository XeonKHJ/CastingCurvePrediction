package xjtuse.castingcurvepredict.data;

public class MlModel {
    private int Id;
    private String Name;
    private String Path;
    private double mLoss = -10;
    private String mStatus = "Created";
    
    public int getId() {
        return Id;
    }

    public void setId(int value) {
        Id = value;
    }

    public String getName() {
        return Name;
    }

    public void setName(String value) {
        Name = value;
    }

    public String getPath() {
        return Path;
    }

    public void setPath(String path) {
        Path = path;
    }

    public void setLoss(double value) {
        mLoss = value;
    }

    public double getLoss() {
        return mLoss;
    }

    public void setStatus(String value) {
        mStatus = value;
    }

    public String getStatus() {
        return mStatus;
    }

}
