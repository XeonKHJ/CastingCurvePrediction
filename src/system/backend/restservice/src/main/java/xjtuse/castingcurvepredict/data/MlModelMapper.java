package xjtuse.castingcurvepredict.data;

import org.apache.ibatis.annotations.Param;

public interface MlModelMapper {
    public MlModel getMlModelById(@Param("id") long id);
}
