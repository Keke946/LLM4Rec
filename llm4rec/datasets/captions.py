"""Strict parser for KuaiRec's four known unquoted caption line breaks."""
from pathlib import Path
import csv
import pandas as pd


def read_kuairec_captions(path: str | Path):
    rows, repairs = [], []
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream)
        header=next(reader)
        if len(header)!=10 or header[:3]!=['video_id','manual_cover_text','caption']:
            raise ValueError('Unexpected official caption schema')
        while True:
            try:row=next(reader)
            except StopIteration:break
            line=reader.line_num
            if len(row)==3 and row[0].isdigit():
                continuation=next(reader,None)
                if continuation is None or len(continuation)!=8:
                    raise ValueError(f'Unrecognized broken caption near line {line}')
                repairs.append(dict(video_id=int(row[0]),first_line=line,last_line=reader.line_num,
                                    repair='join unquoted caption newline; preserve remaining fields'))
                row=[row[0],row[1],row[2]+'\n'+continuation[0],*continuation[1:]]
            if len(row)!=10 or not row[0].isdigit():
                raise ValueError(f'Unexpected caption row at line {line}; refusing to skip data')
            rows.append(row)
    frame=pd.DataFrame(rows,columns=header)
    frame['video_id']=pd.to_numeric(frame.video_id,errors='raise').astype('int64')
    if frame.video_id.duplicated().any():raise ValueError('Duplicate caption video IDs')
    return frame,repairs
