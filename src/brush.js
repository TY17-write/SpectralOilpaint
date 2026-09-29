// Surface contact model. The reference brush remains in shaders.js for comparison.
export function surfaceBrush(nl) {
  return `
  // Strand positions stay fixed in the brush for this stroke, rather than being
  // regenerated for every stamp. Each brush cell retains its own picked-up color.
  let direction=stamp.step.yz;
  var strands=1.;
  if(dot(direction,direction)>.5){
    let across=dot(offset,vec2<f32>(-direction.y,direction.x));
    let at=across*.55;
    let index=i32(floor(at));let t=fract(at);let blend=t*t*(3-2*t);
    let seed=bitcast<u32>(stamp.motion.w)*0x9e3779b9u;
    let wave=mix(randomUnit(bitcast<u32>(index)^seed),randomUnit(bitcast<u32>(index+1)^seed),blend);
    strands=mix(1.,.55+.45*wave,P.v[10].z);
  }
  let contact=m*strands;
  let depth=P.v[10].x+P.v[10].y*stamp.pos.w*stamp.pos.w;
  // Splitting a homogeneous film changes neither its pigment nor its optical
  // composition. Never merge two differently colored layers to make a slot.
  if(n>0u&&n<${nl}u&&q[n-1u].d>depth+eps()){
    let k=n-1u;let fraction=depth/q[k].d;
    q[n]=Layer(depth,q[k].p*fraction,q[k].s,q[k].id);sp[n]=sp[k];
    q[k].d-=depth;q[k].p-=q[n].p;n++;
  }
  // Pressure can progressively bring a little of the next wet layer into reach.
  if(n>1u){
    let k=n-1u;let j=k-1u;
    let lift=min(max(0.,depth-q[k].d),q[j].d)*stepRate(.15,scale)*contact*(1-q[j].s);
    if(lift>0){let pigment=lift*q[j].p/max(q[j].d,1e-30);
      sp[k]=mixwide(sp[k],q[k].p,sp[j],pigment);
      q[k].s=(q[k].s*q[k].d+q[j].s*lift)/max(q[k].d+lift,1e-30);
      q[k].d+=lift;q[k].p+=pigment;q[j].d-=lift;q[j].p-=pigment;
    }
  }
  if(n>0u){
    let k=n-1u;
    // Equal-volume exchange works even with a fully loaded brush. It is bounded
    // by contact depth, not by the total thickness of the paint pile.
    let exchange=stepRate(P.v[10].w,scale)*contact*(1-q[k].s)*min(min(depth,q[k].d),b.d);
    if(exchange>0){
      let fromCanvas=exchange*q[k].p/max(q[k].d,1e-30);
      let fromBrush=exchange*b.p/max(b.d,1e-30);
      let oldCanvas=sp[k];let oldBrush=bWide;
      sp[k]=mixwide(oldCanvas,q[k].p-fromCanvas,oldBrush,fromBrush);
      bWide=mixwide(oldBrush,b.p-fromBrush,oldCanvas,fromCanvas);
      q[k].p+=fromBrush-fromCanvas;b.p+=fromCanvas-fromBrush;
    }
    let pick=min(pickupRate*contact*(1-q[k].s)*min(depth,q[k].d),max(0.,P.v[5].w-b.d));
    let pigment=pick*q[k].p/max(q[k].d,1e-30);
    bWide=mixwide(bWide,b.p,sp[k],pigment);b.d+=pick;b.p+=pigment;
    q[k].d-=pick;q[k].p-=pigment;
  }
  let dep=min(depositRate*contact*b.d,b.d);
  if(dep>0){
    if(n==0u){n=1u;}
    let k=n-1u;
    // Newly deposited paint buries only a matching volume of the previous
    // surface. The rest of the pile is not stirred on each contact.
    if(k>0u){
      let buried=min(min(dep,max(0.,q[k].d+dep-depth)),q[k].d);
      if(buried>0){let pigment=buried*q[k].p/max(q[k].d,1e-30);let j=k-1u;
        sp[j]=mixwide(sp[j],q[j].p,sp[k],pigment);
        q[j].s=(q[j].s*q[j].d+q[k].s*buried)/max(q[j].d+buried,1e-30);
        q[j].d+=buried;q[j].p+=pigment;q[k].d-=buried;q[k].p-=pigment;
      }
    }
    let pigment=dep*b.p/max(b.d,1e-30);
    sp[k]=mixwide(sp[k],q[k].p,bWide,pigment);
    q[k].s=q[k].s*q[k].d/max(q[k].d+dep,1e-30);
    q[k].d+=dep;q[k].p+=pigment;q[k].id=bitcast<u32>(stamp.motion.w);
    b.d-=dep;b.p-=pigment;
  }
  let refill=stepRate(stamp.paint.w,scale)*max(0.,P.v[5].w-b.d);
  bWide=mixwide(bWide,b.p,paint,refill*stamp.motion.z);b.d+=refill;b.p+=refill*stamp.motion.z;
  let packed=narrow(bWide);b.a=packed.a;b.beta=packed.beta;brush[bi]=b;
  for(var k=0u;k<${nl}u;k++){q[k].p=clamp(q[k].p,0.,q[k].d);putlayer(k,i,q[k]);putspec(k,i,narrow(sp[k]));}
  mask[i]=vec4<f32>(contact*drag,contact*drag*stamp.motion.xy,1);
  `;
}
