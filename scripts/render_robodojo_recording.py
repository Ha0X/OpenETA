"""Encode raw and optional same-frame TCP views at simulation speed."""
import argparse
import base64
from bisect import bisect_right
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
LAYOUT=[('cam_head',(0,32,640,480)),('cam_left_wrist',(0,512,320,240)),
        ('cam_right_wrist',(320,512,320,240))]


def tcp_view(raw, row):
    """Project only this frame's public state; missing evidence stays visibly raw."""
    from PIL import Image, ImageDraw, ImageFont
    from tools.robodojo_visual_aids import render_tcp_overlay
    output=raw.copy();draw=ImageDraw.Draw(output)
    font=ImageFont.truetype("DejaVuSans.ttf",10)
    snapshot=row.get('tcp_projection') or {}
    report={'action_index':row['action_index'],'cameras':[]}
    if snapshot.get('status')!='available':
        draw.rectangle((0,0,640,31),fill='black')
        draw.text((8,9),f"Action {row['action_index']} | TCP overlay unavailable (no same-frame data)",fill='white')
        report['status']='unavailable';return output,report
    for name,(x,y,w,h) in LAYOUT:
        frame=deepcopy(snapshot.get('cameras',{}).get(name,{}))
        try:
            ow,oh=frame['width'],frame['height']
            if ow<=0 or oh<=0:raise ValueError('Invalid image dimensions')
            # The raw contact sheet already scaled these camera images. Apply
            # exactly that scale to the intrinsics; do not borrow later poses.
            sx,sy=w/ow,h/oh;intr=frame['intrinsics']
            if intr.get('width',ow)!=ow or intr.get('height',oh)!=oh:
                raise ValueError('Calibration/image mismatch')
            intr.update(fx=intr['fx']*sx,fy=intr['fy']*sy,cx=intr['cx']*sx,cy=intr['cy']*sy,width=w,height=h)
            frame.update(width=w,height=h)
            camera=raw.crop((x,y,x+w,y+h));buffer=BytesIO();camera.save(buffer,format='PNG')
            frame['rgb_base64']=base64.b64encode(buffer.getvalue()).decode()
            encoded,record=render_tcp_overlay(frame,snapshot['tcp_poses'],font=font)
            if encoded:output.paste(Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB'),(x,y))
            else:ImageDraw.Draw(output).text((x+3,y+2),'TCP unavailable: raw view',fill='white',stroke_width=1,stroke_fill='black',font=font)
            report['cameras'].append({'camera':name,'status':record['status'],'reason':record.get('reason')})
        except (KeyError,TypeError,ValueError,ZeroDivisionError):
            ImageDraw.Draw(output).text((x+3,y+2),'TCP unavailable: raw view',fill='white',stroke_width=1,stroke_fill='black',font=font)
            report['cameras'].append({'camera':name,'status':'unavailable','reason':'Invalid same-frame calibration'})
    report['status']='available' if all(c['status']=='available' for c in report['cameras']) else 'partial'
    return output,report


def render(recording, output, fps=25, view='auto'):
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    if view not in ('auto','raw','tcp'):raise ValueError('Unknown recording view')
    rows=[json.loads(line) for line in (recording/'frames.jsonl').read_text().splitlines()]
    if not rows or any(r['simulation_time_s'] is None for r in rows):
        raise ValueError('Recording needs measured simulation timestamps')
    if view=='auto':view='tcp' if any('tcp_projection' in r for r in rows) else 'raw'
    times=[r['simulation_time_s'] for r in rows]
    if times!=sorted(times) or times[0]!=0:raise ValueError('Invalid recording clock')
    duration=times[-1];hold_s=2.;count=max(1,int(np.ceil((duration+hold_s)*fps)))
    first=np.asarray(Image.open(recording/rows[0]['frame']).convert('RGB'))
    writer=imageio_ffmpeg.write_frames(str(output),(first.shape[1],first.shape[0]),fps=fps,
                                      codec='libx264',pix_fmt_out='yuv420p',macro_block_size=1)
    writer.send(None);last_index=None;frame=None;projection_reports=[]
    try:
        for n in range(count):
            index=max(0,bisect_right(times,min(n/fps,duration))-1)
            if index!=last_index:
                current=Image.open(recording/rows[index]['frame']).convert('RGB')
                if view=='tcp':
                    current,report=tcp_view(current,rows[index]);projection_reports.append(report)
                frame=np.asarray(current);last_index=index
            writer.send(frame)
    finally:writer.close()
    report={'source':str(recording.resolve()),'source_frames':len(rows),'fps':fps,'encoded_frames':count,
            'simulation_duration_s':duration,'final_display_hold_s':hold_s,
            'reasoning_pauses_omitted':True,'interpolated_frames':False,
            'sampling':'last measured native-action RGB frame held until next frame',
            'view':view,'tcp_projection_source':'same-frame public state only' if view=='tcp' else None,
            'overlay_note':'Projection may be occluded; not object detection or contact evidence. This continuous recording includes intermediate observations not all sent to the agent.' if view=='tcp' else None,
            'video':str(output.resolve())}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if projection_reports:output.with_suffix('.projection.json').write_text(json.dumps(projection_reports,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--view',choices=['auto','raw','tcp'],default='auto')
    args=parser.parse_args();report=render(args.recording,args.output,view=args.view)
    print(json.dumps(report,indent=2))
    if report['view']=='tcp':
        raw_output=args.output.with_name(args.output.stem+'-raw'+args.output.suffix)
        print(json.dumps(render(args.recording,raw_output,view='raw'),indent=2))
