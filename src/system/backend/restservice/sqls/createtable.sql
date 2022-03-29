CREATE TABLE `castingcurvedb`.`MLModel` (
  `Id` INT NOT NULL AUTO_INCREMENT,
  `Path` VARCHAR(256) NOT NULL,
  `Loss` DOUBLE NULL DEFAULT 2000,
  `Name` VARCHAR(45) NULL DEFAULT 'MLModel',
  `Status` VARCHAR(20) NULL DEFAULT 'Untrained',
  PRIMARY KEY (`Id`),
  UNIQUE INDEX `Path_UNIQUE` (`Path` ASC) VISIBLE)
COMMENT = 'The table that stores ML modes information.';

CREATE TABLE `castingcurvedb`.`Task` (
  `Id` INT NOT NULL AUTO_INCREMENT,
  `Epoch` INT NULL,
  `Loss` DOUBLE NULL,
  `StartTime` DATETIME NULL,
  `StopTime` DATETIME NULL,
  `Status` VARCHAR(20) NULL,
  `ModelId` INT NULL,
  PRIMARY KEY (`Id`),
  INDEX `ModelId_idx` (`ModelId` ASC) VISIBLE,
  CONSTRAINT `ModelId`
    FOREIGN KEY (`ModelId`)
    REFERENCES `castingcurvedb`.`MLModel` (`Id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
