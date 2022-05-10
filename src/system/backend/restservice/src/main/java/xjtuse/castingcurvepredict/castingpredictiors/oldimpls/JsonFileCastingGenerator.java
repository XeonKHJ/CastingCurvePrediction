package xjtuse.castingcurvepredict.castingpredictiors.oldimpls;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import xjtuse.castingcurvepredict.castingpredictiors.GeneratorInput;
import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.models.*;

public class JsonFileCastingGenerator implements ICastingGenerator {

    @Override
    public CastingResultModel PredcitCastingCurve(GeneratorInput input) {
        File file = (File) input.getKeyValues().get("file");
        CastingResultModel resultModel = new CastingResultModel();
        BufferedReader bfReader = null;
        if (file != null && file.canRead()) {
            try {
                bfReader = new BufferedReader(new FileReader(file));
                Boolean firstLine = true;
                String line;
                while ((line = bfReader.readLine()) != null) {
                    if (firstLine) {
                        firstLine = false;
                    } else {
                        String itemToSplit = line;
                        String[] item = itemToSplit.split(",");
                        String date = item[0];
                        double value = Double.parseDouble(item[1]);
                        double liqLevel = 0;
                        if(item.length == 3)
                        {
                            liqLevel = Double.parseDouble(item[2]);
                        }
                            
                        resultModel.addResultItem(date, value, liqLevel);
                    }
                }
            } catch (NumberFormatException | IOException e) {
                // TODO 开浇曲线预测会出现的错误。
                e.printStackTrace();
            }
        }

        return resultModel;
    }

    @Override
    public void updateModel(CastingModel data) {
        // TODO Auto-generated method stub

    }

    @Override
    public void updateModel(ArrayList<CastingModel> datas) {
        // TODO Auto-generated method stub

    }

    @Override
    public void train(TaskModel task) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void stop(TaskModel task) {
        // TODO Auto-generated method stub
        
    }
}
