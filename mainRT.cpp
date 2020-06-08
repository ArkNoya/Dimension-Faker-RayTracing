/*
        Dimension Faker v1.1

        updata:
            1.render channal (render pass)
                Diffuse,Specular,Reflaction,Refraction,Emission
            2.add Emission Shade
            3.add easy sun light
            4.add drand48 lib
*/

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <ctime>
using namespace std;

#define UC unsigned char
#define pia 3.14159
#define rad1 .0174532
#define BSURP vec surp = hi.pos + hi.N*.00003

enum channal {ALL,DIF,SPC,RFL,RFR,EMT};



#ifndef DRAND48_H
#define DRAND48_H
/*
https://www.cnblogs.com/yoyo-sincerely/p/8854922.html
*/
#define drand48m 0x100000000LL  
#define drand48c 0xB16  
#define drand48a 0x5DEECE66DLL  
unsigned long long seed = 1;
float drand48(void)
{
	seed = (drand48a * seed + drand48c) & 0xFFFFFFFFFFFFLL;
	unsigned int x = seed >> 16;
	return  ((float)x / (float)drand48m);

}
void srand48(unsigned int i)
{
	seed = (((long long int)i) << 16) | rand();
}
#endif



struct vec
{
    float x,y,z;

    vec(float i=.0):
        vec(i,i,i) {}
    vec(float x, float y, float z):
        x(x), y(y), z(z) {}
};
struct vec2
{
    float u,v;

    vec2(float i=.0):
        vec2(i,i) {}
    vec2(float u, float v):
        u(u), v(v) {}
};

vec operator+(const vec & v0, const vec & v1){
    return vec(v0.x + v1.x,
                v0.y + v1.y,
                v0.z + v1.z
    );
}
vec operator-(const vec & v0, const vec & v1){
    return vec(v0.x - v1.x,
                v0.y - v1.y,
                v0.z - v1.z
    );
}
vec operator*(const vec & v0, const vec & v1){
    return vec(v0.x * v1.x,
                v0.y * v1.y,
                v0.z * v1.z
    );
}
vec operator/(const vec & v0, const vec & v1){
    return vec(v0.x / v1.x,
                v0.y / v1.y,
                v0.z / v1.z
    );
}

vec operator+=(vec & v0, const vec & v1){
    v0 = v0 + v1;
    return v0;
}
vec operator/=(vec & v0, const vec & v1){
    v0 = v0 / v1;
    return v0;
}

vec max(const vec & v0, const vec & v1){
    return vec(
        max(v0.x,v1.x),
        max(v0.y,v1.y),
        max(v0.z,v1.z)
    );
}

float dot(const vec & v0, const vec & v1){
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}
float length(const vec & v){
    return sqrtf(dot(v,v));
}
vec normalize(const vec & v){
    return v/length(v);
}
vec cross(const vec & v0, const vec & v1){
    return vec(
        v0.y*v1.z - v0.z*v1.y,
        v0.z*v1.x - v0.x*v1.z,
        v0.x*v1.y - v0.y*v1.x
    );
}
float angle(const vec & v0, const vec & v1){
    return acosf(dot(normalize(v0),normalize(v1)));
}

vec randVec(){
    vec v;
    do
    {
        v = vec(drand48(),drand48(),drand48());
        v = v*2-1;
    } while (length(v)>1);
    
    return v;
}

float clamp(const float & f0, const float & omin, const float & omax){
    float ff = f0;

    if(ff>omax) ff = omax;
    if(ff<omin) ff = omin;

    return ff;
}
float clamp01(const float & f){
    return clamp(f,0,1);
}

vec clamp(const vec & v, const vec & vmin, const vec & vmax){
    return vec(
        clamp(v.x,vmin.x,vmax.x),
        clamp(v.y,vmin.y,vmax.y),
        clamp(v.z,vmin.z,vmax.z)
    );
}
vec clamp01(const vec & v){
    return clamp(v,0,1);
}

struct Ray
{
    vec org,dir;
    Ray(vec org = 0, vec dir = 0):
        org(org), dir(normalize(dir)) {}
    vec rich(const float & t)const{
        return org + dir*t;
    }
};

struct Camera
{
    int cw, ch;
    float rh, ratio, rw;
    vec org, lookAt, up;
    float fov;
    vec dir, rdir, sup;

    Camera(int cw, int ch, vec org = 5, vec lookAt = 0, float fov = 50, vec up = vec(0,1,0))
        :cw(cw), ch(ch), org(org), lookAt(lookAt), fov(fov), up(up) {
            dir = lookAt - org;
            rdir = normalize(cross(dir,up));
            sup = normalize(cross(rdir,dir));
            ratio = 1.0*cw/ch;
            rh = tanf(fov/2*rad1);
            rw = rh*ratio;
        }
    
    Ray camRay(float ix, float iy){
        vec npc = org + normalize(dir);
        float ox = ix/cw*2-1;
        float oy = iy/ch*2-1;
        vec shutPos = npc + sup*oy*rh + rdir*ox*rw;

        return Ray(org,shutPos-org);
    }
};

struct texture;
struct material;

struct hitInfo
{
    float t;
    vec pos,N;
    material * himat;

    hitInfo(material * himat, float t = INFINITY, vec pos = 0, vec N = vec(0,1,0)):
        t(t), pos(pos), N(normalize(N)), himat(himat) {}
};

struct texture {
	virtual vec value(const vec2 & uv, const hitInfo & hi, const Ray & ray)const = 0;
};

struct flat : texture {
	vec color;

	flat(float r, float g, float b)
		:color(vec(r, g, b)) {}
	flat(vec color = .8) : color(color) {}

	vec value(const vec2 & uv, const hitInfo & hi, const Ray & ray)const {
		return color;
	}
};

struct checkBoard : texture {
	vec color;
    vec darkColor;
	float sx, sy, sz;
    bool contrast;

    checkBoard(float r, float g, float b, float dr, float dg, float db, float sx = 2, float sy = 2, float sz = 2, bool contrast = true)
		:color(vec(r,g,b)), darkColor(vec(dr,dg,db)), sx(sx), sy(sy), sz(sz), contrast(contrast) {}
	checkBoard(vec color = .8, vec darkColor = .4, float sx = 2, float sy = 2, float sz = 2, bool contrast = true)
		:color(color), darkColor(darkColor), sx(sx), sy(sy), sz(sz), contrast(contrast) {}

	vec value(const vec2 & uv, const hitInfo & hi, const Ray & ray)const {
		float lum0 = sin(hi.pos.x*sx*(pia/2));
		float lum1 = sin(hi.pos.z*sz*(pia/2));
        float ol = lum0*lum1/2 + .5;
        if(contrast) ol = ol>.5 ? 1:0;
		
		return max(color*ol, darkColor*(1-ol));
	}
};
struct toonEdge : texture {
	vec color;
	float size;
	float alpha;
	bool contrast;

    toonEdge(float r, float g, float b, float size = 15, float alpha = 1)
		:color(vec(r,g,b)), size(size), alpha(alpha), contrast(contrast) {}
	toonEdge(vec color = .8, float size = 15, float alpha = 1)
		:color(color), size(size), alpha(alpha), contrast(contrast) {}

	vec value(const vec2 & uv, const hitInfo & hi, const Ray & ray)const {
		float lum0 = angle(ray.dir*-1, hi.N);
		lum0 = lum0 > (90 - size)*rad1 ? (1-alpha) : 1;

		return color * lum0;
	}
};
struct cosRing : texture {
	vec color;
    vec multColor;
	vec center;
	float scale, offset;
    bool contrast;

    cosRing(float r, float g, float b, float mr, float mg, float mb, vec center = vec(1.33,0,0), float scale = 12, float offset = 0, bool contrast = true)
		:color(vec(r,g,b)), multColor(vec(mr,mg,mb)), center(center), scale(scale), contrast(contrast) {}
    
    cosRing(float r, float g, float b, vec multColor = .4, vec center = vec(1.33,0,0), float scale = 12, float offset = 0, bool contrast = true)
		:color(vec(r,g,b)), multColor(multColor), center(center), scale(scale), contrast(contrast) {}
	
    cosRing(vec color = .8, vec multColor = .4, vec center = vec(1.33,0,0), float scale = 12, float offset = 0, bool contrast = true)
		:color(color), multColor(multColor), center(center), scale(scale), contrast(contrast) {}

	vec value(const vec2 & uv, const hitInfo & hi, const Ray & ray)const {
		float mult = cos(length(hi.pos - center)*scale+offset)/2 + .5;
        if(contrast) mult = mult>.5 ? 1:0;
		return max(color*mult,multColor*(1-mult));
	}
};
struct normalFlat : texture {

	vec value(const vec2 & uv, const hitInfo & hi, const Ray & ray)const {
		return hi.N;
	}
};

struct scatterInfo
{
    Ray outRay;
    vec mult;

    scatterInfo(Ray outRay = Ray(), vec mult = 1):
        outRay(outRay), mult(mult) {}
};

struct material
{
    virtual scatterInfo scatter(const hitInfo & hi, const Ray & ray)const=0;
    virtual channal getType()const=0;
};

struct Diffuse:material
{
    texture * color;

    Diffuse(texture * color):
        color(color) {}
    
    scatterInfo scatter(const hitInfo & hi, const Ray & ray)const{
        vec nv = hi.N + randVec();
        BSURP;
        Ray oRay(surp,nv);
        vec mult = dot(oRay.dir,hi.N);
        vec gcolor = color->value(0,hi,ray);

        return scatterInfo(oRay,mult*gcolor);
    }

    channal getType()const{
        return DIF;
    }
};

vec reflaction(const vec & N, const vec & inv, const float & roughness) {
	float sn = dot(inv, N*-1);
	vec rv = inv + N * 2.*sn;

	if (roughness > 0) {
        float maxr = dot(N, rv);
		float nrough = roughness;
		if (roughness > maxr) {
			nrough = maxr;
		}
		rv += randVec()*nrough;
	}

	return rv;
}
struct Metal:material{
    texture * color;
    float Rounghness;

    Metal(texture * color, float Rounghness = .05)
        :color(color), Rounghness(Rounghness) {}
    
    scatterInfo scatter(const hitInfo & hi, const Ray & ray)const{
        vec ov = reflaction(hi.N,ray.dir,Rounghness);
        vec gcolor = color->value(0,hi,ray);
        BSURP;
        return scatterInfo(Ray(surp,ov), gcolor);
    }

    channal getType()const{
        return RFL;
    }
};

vec refraction(const vec & N, const vec & inv, const float & ior, const float & roughness, int & dirs ) {
	dirs = 1;
	if (dot(N, inv) < 0) {
		float ia = angle(N, inv*-1);
		float oa = asin(sin(ia) / ior);
		vec dir = normalize(cross(N, cross(inv, N)));
		vec nv = dir * sin(oa) - N * cos(oa);

		if (roughness > 0) {
			float maxr = cos(oa);
			float nrough = roughness;
			if (roughness > maxr)
				nrough = maxr;
			nv = nv + randVec()*nrough;
		}
		return nv;
	}
	else {
		float ia = angle(N, inv);
		float oa = asin(sin(ia)*ior);
		float ca = asin(1 / ior);

		if (ia > ca) {
			dirs = -1;
			return reflaction(N, inv, roughness);
		}

		vec dir = normalize(cross(N, cross(inv, N)));
		vec nv = dir * sin(oa) + N * cos(oa);
			
		if (roughness > 0) {
			float maxr = cos(oa);
			float nrough = roughness;
			if (roughness > maxr)
				nrough = maxr;
			nv = nv + randVec()*nrough;
		}
		return nv;
	}
}
struct snell : material{
	texture * color;
	float ior;
	float roughness;

	snell(texture * color, float ior = 1.33, float roughness = .0)
		:color(color), ior(ior), roughness(roughness) {}

	scatterInfo scatter(const hitInfo & hi, const Ray & ray)const {
		int dirs = 1;
		vec nv = refraction(hi.N, ray.dir, ior, roughness, dirs);
		Ray nr(hi.pos + ray.dir*.00003*dirs, nv);
        vec gcolor = color->value(0,hi,ray);
		return scatterInfo(nr, gcolor);
	}

    channal getType()const{
        return RFR;
    }
};

struct Emission:material
{
    texture * color;
    float scale;

    Emission(texture * color, float scale = 2):
        color(color), scale(scale) {}
    
    scatterInfo scatter(const hitInfo & hi, const Ray & ray)const{
        vec gcolor = color->value(0,hi,ray);
        return scatterInfo(Ray(),gcolor*scale);
    }

    channal getType()const{
        return EMT;
    }
};


material * globalMat = new Diffuse(new flat);

struct obj
{
    virtual bool hit(hitInfo & hi, const Ray & ray)const=0;
};

struct Sphere:obj
{
    vec cen;
    float rad;
    material * objmat;

    Sphere(vec cen = vec(0,1.8,0), float rad = 1.8, material * objmat = globalMat):
        cen(cen), rad(rad), objmat(objmat) {}
    
    bool hit(hitInfo & hi, const Ray & ray)const{
        vec oc = ray.org - cen;
        float b = 2*dot(oc,ray.dir);
        float c = dot(oc,oc) - rad*rad;
        float delta = b*b - 4*c;

        if(delta<0) return false;

        float t = (-b-sqrtf(delta))/2;
        if(length(oc) < rad) t = (-b+sqrtf(delta))/2;
        if(t<0) return false;

        hi.t = t;
        vec pos = ray.rich(t);
        hi.pos = pos;
        hi.N = normalize(pos-cen);
        hi.himat = objmat;
        return true;
    }
};

struct obj_list:obj
{
    obj ** list;
    int size;

    obj_list(obj ** ol, int s):
        list(ol), size(s) {}
    
    bool hit(hitInfo & hi, const Ray & ray)const{
        bool fb = list[0]->hit(hi,ray);

        if(size<=1) return fb;

        for(int i=0; i<size-1; i++){
            hitInfo nhi(globalMat);
            bool nb = list[i+1]->hit(nhi,ray);
            if(!nb) continue;
            if(nhi.t < hi.t){
                hi = nhi;
                fb = nb;
            }
        }
        return fb;
    }
};

vec shading(const obj * scenes, const Ray & ray, const int & rayDepth, const channal & rc, bool isfst = true){
    
    if(rayDepth<=0) return 0;

    hitInfo hi(globalMat);
    bool ishit = scenes->hit(hi,ray);
    if(ishit){

        channal cc = hi.himat->getType();


        if((cc==EMT && !isfst) || (cc==EMT && rc==EMT && isfst)){
            scatterInfo si = hi.himat->scatter(hi,ray);
            return si.mult;
        }

        if(rc==cc || !isfst){
            scatterInfo si = hi.himat->scatter(hi,ray);
            return shading(scenes,si.outRay,rayDepth-1,rc,false)*si.mult;
        }else{
            return 0;
        }
    }else{
            float lum = angle(ray.dir,vec(0,1,0))/pia;
            float sunMult = 1-clamp01(angle(vec(-.15,1,-1),ray.dir)/rad1/12);
            return (1-lum)*vec(.55,.65,.9999) + lum*vec(.999,.85,.8) + sunMult*vec(1,.75,.6)*30;
    }
}

vec atangamma(const vec & v){
    return vec(
        atanf(v.x*(pia/2)),
        atanf(v.y*(pia/2)),
        atanf(v.z*(pia/2))
    );
}

struct BMHead
{
    UC BM[2] = {0x42,0x4d};
    UC BMs[4];
    UC keepArea[4] = {0x00,0x00,0x00,0x00};
    UC sa[4] = {0x36,0x00,0x00,0x00};
    UC hs[4] = {0x28,0x00,0x00,0x00};
    UC Bw[4];
    UC Bh[4];
    UC dl[2] = {0x01,0x00};
    UC cb[2] = {0x18,0x00};
    UC nullA[24];

    BMHead(int w, int h){
        int fs = w*h*3 + 36;
        for(int i=0; i<4; i++){
            BMs[i] = (UC)(fs%256);
            fs /= 256;
        }

        for(int i=0; i<4; i++){
            Bw[i] = (UC)(w%256);
            w /= 256;
        }
        for(int i=0; i<4; i++){
            Bh[i] = (UC)(h%256);
            h /= 256;
        }
        for(int i=0; i<24; i++){
            nullA[i]=0x00;
        }
    }

    UC operator[](const int & i)const {
        return BM[i];
    }
    int size()const {
        return 54;
    }
};

void show(vec v){
    cout<<"("<<v.x<<","<<v.y<<","<<v.z<<")";
}

void BMofs(ofstream & ofs, const vec & color){
    ofs<<(UC)(clamp01(color.z)*255)<<(UC)(clamp01(color.y)*255)<<(UC)(clamp01(color.x)*255);
}

int main(){

    int w = 1920;
    int h = 1080;
    float scale = 1;
    w*=scale;
    h*=scale;
    cout<<"w:"<<w<<" h:"<<h<<endl;

    //create directory
    system("md .\\outImage");

    // .bmp file create
    ofstream ofs(".\\outImage\\beauty.bmp", ios::binary);
    ofstream ofsd(".\\outImage\\Diffuse.bmp", ios::binary);
    ofstream ofss(".\\outImage\\Specular.bmp", ios::binary);
    ofstream ofsl(".\\outImage\\Reflaction.bmp", ios::binary);
    ofstream ofsr(".\\outImage\\Refraction.bmp", ios::binary);
    ofstream ofse(".\\outImage\\Emission.bmp", ios::binary);
    BMHead bmh(w,h);
    for(int i=0; i<bmh.size(); i++){
        ofs<<bmh[i];
    }
    for(int i=0; i<bmh.size(); i++){
        ofsd<<bmh[i];
    }
    for(int i=0; i<bmh.size(); i++){
        ofss<<bmh[i];
    }
    for(int i=0; i<bmh.size(); i++){
        ofsl<<bmh[i];
    }
    for(int i=0; i<bmh.size(); i++){
        ofsr<<bmh[i];
    }
    for(int i=0; i<bmh.size(); i++){
        ofse<<bmh[i];
    }

    //obj build scenes
    Camera cam0(w,h,vec(.15,2.3,4.5),vec(.11,.8,0),66);
    obj * list[9];
    list[0] = new Sphere(vec(0,-200,0),200, new Diffuse(new checkBoard(vec(.8),vec(.3),3,2,3)));
    list[1] = new Sphere(vec(.05,1.11,0),1.1, new Diffuse(new toonEdge(.7,.8,.99,15)));
    list[2] = new Sphere(vec(3.41,2.51,0),2.5,new Metal(new flat(.9,.6,.5),.09));
    list[3] = new Sphere(vec(-2.91,1.91,0),1.9,new Metal(new flat(.92,.75,.6),.028));
    list[4] = new Sphere(vec(2.1, 1.501, 4.5), 1.2, new Diffuse(new flat(.5, .95, .6)));
    list[5] = new Sphere(vec(-.4, .71, .7 + 1.1), .7, new snell(new flat(.89, .91, .95), 1.45, .015));
    list[6] = new Sphere(vec(.45, .81, -.81 - .68), .8, new snell(new flat(.98), 1.05));
    list[7] = new Sphere(vec(-1.8, 1.301, 3.8), 1.1, new Diffuse(new cosRing(vec(.95, .52, .5),vec(.2),0)));
    list[8] = new Sphere(vec(1.5,.3,1.7),.2, new Emission(new flat(.98,.8,.7), 2.5));
    obj * scenes = new obj_list(list,9);

    //render options
    int sampleTimes = 6;
    int rayDepth = 6;
    channal renderChannal = ALL;
    //user interface
    cout<<"Enter sampleTime:";cin>>sampleTimes;cout<<"Enter rayDepth:";cin>>rayDepth;cout<<endl<<"0.ALL\n1.Diffuse\n2.Specular\n3.Reflaction\n4.Refracion\n5.Emission"<<endl<<"Enter render chanal:";int getrc;cin>>getrc;switch(getrc){case 0:renderChannal=ALL;break;case 1:renderChannal=DIF;break;case 2:renderChannal=SPC;break;case 3:renderChannal=RFL;break;case 4:renderChannal=RFR;break;case 5:renderChannal=EMT;break;};

    //color list
    vector<vec> colorsDIF(w*h);
    vector<vec> colorsSPC(w*h);
    vector<vec> colorsRFL(w*h);
    vector<vec> colorsRFR(w*h);
    vector<vec> colorsEMT(w*h);

    //render
    int t0 = time(NULL);
    cout<<"rendering..."<<endl;
    for(int j=0; j<5; j++){
        int uj;
        if(renderChannal==ALL){
            uj = j+1;
        }else{
            uj = renderChannal;
            j=5;
        }
#pragma omp parallel for
        for(int i=0; i<w*h; i++){
            int x = i%w;
            int y = i/w;
            
            vec color;
            for(int ii=0; ii<sampleTimes; ii++){
                Ray cray = cam0.camRay(x+(drand48()*2-1)*.5,y+(drand48()*2-1)*.5);

                hitInfo hi(globalMat);

                color += shading(scenes,cray,rayDepth,channal(uj));
            }
            color /= sampleTimes;
            switch (channal(uj)) 
            {
            case DIF:
                colorsDIF[i] = color;
                break;
            case SPC:
                colorsSPC[i] = color;
                break;
            case RFL:
                colorsRFL[i] = color;
                break;
            case RFR:
                colorsRFR[i] = color;
                break;
            case EMT:
                colorsEMT[i] = color;
                break;
            default:
                break;
            }
        }
    }
    int t1 = time(NULL);
    cout<<"rendering completed ! "<<"time used : "<<t1-t0<<" seconds"<<endl;

    //writing files
    t0 = time(NULL);
    cout<<"wirting file..."<<endl;
    //mix all color
    vector<vec> colors(w*h);
#pragma omp parallel for
    for(int i=0; i<w*h; i++){
        colors[i] = max(max(max(max(colorsDIF[i],colorsSPC[i]),colorsRFL[i]),colorsRFR[i]),colorsEMT[i]);
    }
    //writing
#pragma omp parallel for
    for (int i = 0; i < 6; i++)
    {
        switch (i)
        {
        case 0:
            for(int i=0; i<w*h; i++){
                BMofs(ofs,atangamma(colors[i]));
            }
            break;
        case 1:
            for(int i=0; i<w*h; i++){
                if(renderChannal!=ALL && renderChannal!=DIF) break;
                BMofs(ofsd,atangamma(colorsDIF[i]));
            }
            break;
        case 2:
            for(int i=0; i<w*h; i++){
                if(renderChannal!=ALL && renderChannal!=SPC) break;
                BMofs(ofss,atangamma(colorsSPC[i]));
            }
            break;
        case 3:
            for(int i=0; i<w*h; i++){
                if(renderChannal!=ALL && renderChannal!=RFL) break;
                BMofs(ofsl,atangamma(colorsRFL[i]));
            }
            break;
        case 4:
            for(int i=0; i<w*h; i++){
                if(renderChannal!=ALL && renderChannal!=RFR) break;
                BMofs(ofsr,atangamma(colorsRFR[i]));
            }
            break;
        case 5:
            for(int i=0; i<w*h; i++){
                if(renderChannal!=ALL && renderChannal!=EMT) break;
                BMofs(ofse,atangamma(colorsEMT[i]));
            }
            break;

        default:
            break;
        }
    }
    t1 = time(NULL);
    cout<<"writing file completed ! "<<"time used : "<<t1-t0<<" seconds"<<endl;
    cout<<"write file completed !"<<endl;

    ofs.close();
    ofsd.close();
    ofss.close();
    ofsl.close();
    ofsr.close();
    ofse.close();
    switch (renderChannal)
    {
    case ALL:
        system(".\\outImage\\beauty.bmp");
        break;
    case DIF:
        system(".\\outImage\\Diffuse.bmp");
        break;
    case SPC:
        system(".\\outImage\\Specular.bmp");
        break;
    case RFL:
        system(".\\outImage\\Reflaction.bmp");
        break;
    case RFR:
        system(".\\outImage\\Refraction.bmp");
        break;
    case EMT:
        system(".\\outImage\\Emission.bmp");
        break;
    
    default:
        break;
    }

    return 233;
}
