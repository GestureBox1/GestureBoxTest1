
export interface IData {
    label: ILabels;
    data: Array<IDataValues>;
}
export interface ILabels {
    label_x: string;
    label_y: string;
    label_z: string;
}
export interface IDataValues {
    data_z: string;
    data_x: Array<number>;
    data_y: Array<number>;
}