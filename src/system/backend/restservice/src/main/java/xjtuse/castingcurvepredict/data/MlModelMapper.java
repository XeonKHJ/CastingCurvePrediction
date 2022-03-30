package xjtuse.castingcurvepredict.data;

import java.util.List;

import org.apache.ibatis.annotations.Param;

public interface MlModelMapper {
    public MlModel getMlModelById(@Param("id") long id);
    public List<MlModel> getModels();
}
