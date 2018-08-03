
    
    
export interface IPadding {
    left?:number;
    right?:number;
    bottom?:number;
    top?:number;
}
export interface ISettings {
        id: string;
        svgHeight?: number;
        svgWidth?: number;
        padding? :IPadding;
        pathColors? : Array<string>;
        dataPointLimit? : number;
        defaultScale? : number;
        xAxisUnit? : string;
        zAxisUnit? : string;
        showLabels? : boolean;
        showAxisUnits? : boolean;
}