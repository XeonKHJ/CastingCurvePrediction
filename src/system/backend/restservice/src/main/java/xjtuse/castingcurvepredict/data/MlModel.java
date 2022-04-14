package xjtuse.castingcurvepredict.data;

public class MlModel {
    private int Id;
    private String Name;
    private String Path;
    private double mLoss;

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

    public void setLoss(double value)
    {
        mLoss = value;
    }

    public double getLoss()
    {
        return mLoss;
    }

}
