#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <drand48.h>
#include <time.h>
using namespace std;
#define uchar unsigned char
#define uint unsigned int
#define pia 3.14159265358979323
#define rad1 .01745329251994329

struct vec {
	double x, y, z;

	vec(double i = .0)
		: vec(i, i, i) {}
	vec(double x, double y, double z)
		: x(x), y(y), z(z) {}
};
struct vec_c {
	uchar r, g, b;

	vec_c(uchar i = 0x00)
		: vec_c(i,i,i) {}
	vec_c(uchar r, uchar g, uchar b)
		: r(r), g(g), b(b) {}
};
vec operator+(const vec & a, const vec & b) {
	return vec(a.x + b.x,
		a.y + b.y, a.z + b.z);
}
vec operator+=(vec & a, const vec & b) {
	a = a + b;
	return a;
}
vec operator-(const vec & a, const vec & b) {
	return vec(a.x - b.x,
		a.y - b.y, a.z - b.z);
}
vec operator*(const vec & a, const vec & b) {
	return vec(a.x*b.x,
		a.y*b.y, a.z*b.z);
}
vec operator*=(vec & a, const vec & b) {
	a = a * b;
	return a;
}
vec operator/(const vec & a, const vec & b) {
	return vec(a.x / b.x,
		a.y / b.y, a.z / b.z);
}
vec operator/=(vec & a, const vec & b) {
	a = a / b;
	return a;
}

vec cross(const vec & a, const vec & b) {
	return vec(a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x);
}
double dot(const vec & a, const vec & b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
double length(const vec & v) {
	return sqrt(dot(v, v));
}
vec normalize(const vec & v) {
	return v / length(v);
}
double distance(const vec & a, const vec & b) {
	return length(a - b);
}
double angle(const vec & a, const vec & b) {
	return acos(dot(normalize(a), normalize(b)));
}

struct vec2 {
	double x, y;

	vec2(double x, double y)
		: x(x), y(y) {}
};

ostream & operator<<(ostream & os, vec v) {
	os << "(" << v.x << "," << v.y << "," << v.z << ")";
	return os;
}

double efit(const double i,
	const double & imin, const double & imax,
	const double & omin, const double & omax) {
	double scale =
		abs(omax - omin) /
		abs(imax - imin);

	int id = imax > imin ? 1 : -1;
	int od = omax > omin ? 1 : -1;
	int sd = i > imin ? 1 : -1;

	double dis = abs(i - imin);
	double odis = dis * scale;

	return omin + odis * id*od*sd;
}
double fit(const double & i,
	const double & imin,const double & imax,
	const double & omin, const double & omax) {
	if (i > imax)
		return omax;
	if (i < imin)
		return omin;
	return efit(i, imin, imax,
		omin, omax);
}
double fit01(const double & i,
	const double & omin, const double & omax) {
	return fit(i, 0, 1, omin, omax);
}
vec fit(const vec & v,
	const double & imin,const double & imax,
	const double & omin,const double & omax) {
	return vec(fit(v.x, imin,
		imax, omin, omax),
		fit(v.y, imin,
			imax, omin, omax),
		fit(v.z, imin,
			imax, omin, omax));
}
vec fit01(const vec & v,
	const double & omin, const double & omax) {
	return fit(v, 0, 1, omin, omax);
}

vec randvec() {
	return normalize(vec(fit01(drand48(), -1, 1),
		fit01(drand48(), -1, 1),
		fit01(drand48(), -1, 1)));
}

struct Ray {
	vec org, dir;

	Ray(vec org, vec dir)
		:org(org), dir(normalize(dir)) {}
};

struct material;
struct hitInfo {
	bool isHit;
	double t;
	vec hitPos, N;
	material * ptmat;

	hitInfo(bool isHit, double t, vec hitPos, vec N, material * ptmat)
		:isHit(isHit), t(t), hitPos(hitPos), N(normalize(N)), ptmat(ptmat) {}
};
struct scatterInfo {
	Ray outRay;
	vec mult;

	scatterInfo(Ray oray, vec m)
		: outRay(oray), mult(m) {}
};
struct material {
	virtual scatterInfo scatter(const hitInfo & hi, const Ray & ray)const = 0;
};

struct diffuse : material {
	vec color;

	diffuse(vec c = vec(.8))
		:color(c) {}

	scatterInfo scatter(const hitInfo & hi, const Ray & ray)const {
		vec rv = hi.N + randvec();
		vec mult = color * cos(angle(hi.N, rv));
		Ray nr(hi.hitPos + hi.N*1e-12, rv);
		return scatterInfo(nr, mult);
	}
};

vec reflaction(vec iv, vec n) {
	return iv + n * abs(dot(iv, n)) * 2;
}
struct metal : material {
	vec color;
	double roughness;

	metal(vec c, double r = .01)
		:color(c), roughness(r) {}

	scatterInfo scatter(const hitInfo & hi, const Ray & ray)const {
		vec nv = reflaction(ray.dir, hi.N);
		vec rv = randvec()*roughness + nv;
		if (angle(rv, hi.N) > 90 * rad1) {
			vec di = normalize(cross(hi.N, cross(ray.dir, hi.N)));
			double clamproughness = asin(angle(di, nv));
			rv = randvec()*clamproughness + nv;
		}
		return scatterInfo(Ray(hi.hitPos + hi.N*1e-12, rv), color);
	}
};

vec refraction(const vec & iv,const vec & n, const double & ior, int & ds) {

	if (angle(iv, n) > 90 * rad1) {
		ds = 1;
		double ia = angle(iv*-1, n);
		double oa = asin(sin(ia) / ior);
		vec dir = normalize(cross(n, cross(iv, n)));
		vec ov = dir * sin(oa) + n * -1 * cos(oa);
		return ov;
	}
	else {
		double ia = angle(iv, n);
		double oa = asin(sin(ia)*ior);
		double ca = asin(1 / ior);
		vec dir = normalize(cross(n, cross(iv, n)));
		if (ia > ca) {
			ds = -1;
			return reflaction(iv, n*-1);
		}
		vec ov = dir * sin(oa) + n * cos(oa);
		return ov;
	}
}
struct snell : material {
	vec color;
	double ior;
	double roughness;

	snell(vec color, double ior = 1.33, double roughness = .01)
		:color(color), ior(ior), roughness(roughness) {}

	scatterInfo scatter(const hitInfo & hi, const Ray & ray)const {
		int ds = 1;
		vec ov = refraction(ray.dir, hi.N, ior, ds);
		vec rv = ov + randvec()*roughness;
		if (angle(rv, hi.N) > 90 * rad1)
			rv += ov;
		return scatterInfo(Ray(hi.hitPos + ray.dir*ds*1e-12, rv), color);
	}
};

struct gridDif : material {
	vec color;
	double sx, sy, sz;
	double maxl, minl;

	gridDif(vec color, double sx = 2., double sy = 2., double sz = 2.,
					double minl = 1., double maxl = .399)
		:color(color), sx(sx), sy(sy), sz(sz), maxl(maxl), minl(minl) {}

	scatterInfo scatter(const hitInfo & hi, const Ray & ray)const {
		double lx = (int)round(hi.hitPos.x*sx) % 2 == 0 ? 1 : 0;
		double ly = (int)round(hi.hitPos.y*sy) % 2 == 0 ? 1 : 0;
		double lz = (int)round(hi.hitPos.z*sz) % 2 == 0 ? 1 : 0;
		double ml = fit(lx * ly*lz, 0, 1, minl, maxl);

		vec rv = randvec() + hi.N;
		double m2 = cos(angle(rv, hi.N));
		return scatterInfo(Ray(hi.hitPos + hi.N*1e-12, rv), color*ml*m2);
	}
};

struct toonDif : material {
	vec color;
	double edge;

	toonDif(vec color, double edge = 15)
		:color(color), edge(edge) {}

	scatterInfo scatter(const hitInfo & hi, const Ray & ray)const {
		vec rv = hi.N + randvec();
		double mt = cos(angle(hi.N, rv));

		double mt2 = angle(hi.N, ray.dir*-1) / rad1;
		if (mt2 > 90 - edge)
			mt2 = 0;
		else mt2 = 1;

		return scatterInfo(Ray(hi.hitPos + hi.N*1e-12, rv), color*mt*mt2);
	}
};

struct cosGridDif : material {
	vec color;
	vec waveCenter;
	double scale;
	double min, max;
	int constant;

	cosGridDif(vec color, vec waveCenter, double scale = 17, double min = 0, double max = 1, int constant = 1)
		: color(color), waveCenter(waveCenter), scale(scale), min(min), max(max), constant(constant) {}

	scatterInfo scatter(const hitInfo & hi, const Ray & ray)const {
		vec rv = randvec() + hi.N;
		double m0 = cos(angle(hi.N, rv));
		
		double m1 = fit(cos(distance(waveCenter, hi.hitPos)*scale), -1, 1, min, max);
		if (constant) {
			m1 = m1 > .5 ? 1 : 0;
		}

		return scatterInfo(Ray(hi.hitPos + hi.N*1e-12, rv), color*m0*m1);
	}
};

struct obj {
	virtual hitInfo hit(const Ray & ray)const = 0;
};
struct Sphere : obj {
	vec cen;
	double rad;
	material * mate;

	Sphere(vec cen, double rad, material * mate)
		:cen(cen), rad(rad), mate(mate) {}

	hitInfo hit(const Ray & ray)const {
		vec oc = ray.org - cen;
		double b = 2. * dot(oc, ray.dir);
		double c = dot(oc, oc) - rad * rad;
		double delta = b * b - 4.*c;

		if(delta < 0)
			return hitInfo(false, INFINITY, ray.org, vec(0), mate);

		double t = (-b - sqrt(delta)) / 2;
		if(distance(ray.org, cen) > rad && t < 0)
			return hitInfo(false, INFINITY, ray.org, vec(0), mate);
		if (distance(ray.org, cen) < rad)
			t = (-b + sqrt(delta)) / 2;

		vec hp = ray.org + ray.dir*t;
		vec n = hp - cen;

		return hitInfo(true, t, hp, n, mate);
	}
};
struct Ground : obj {
	vec cen, N;
	material * mate;

	Ground(vec c, vec N, material * mate)
		:cen(c), N(normalize(N)), mate(mate) {}

	hitInfo hit(const Ray & r)const {
		vec oc = r.org - cen;
		double t = dot(N, oc)*-1 / dot(N, r.dir);

		if (t < 0)
			return hitInfo(false, INFINITY, r.org, vec(0), mate);

		vec hp = r.org + r.dir*t;

		return hitInfo(true, t, hp, N, mate);
	}
};
struct obj_list : obj {
	obj ** list;
	int size;

	obj_list(obj ** ol, int s)
		: list(ol), size(s) {}

	hitInfo hit(const Ray & r)const {
		hitInfo fhi = list[0]->hit(r);
		for (int i = 0; i < size - 1; i++) {
			hitInfo chi = list[i + 1]->hit(r);
			if (chi.t < fhi.t)
				fhi = chi;
		}
		return fhi;
	}
};

struct Camera {
	vec org;
	vec lookAt;
	double fov;
	vec up;
protected:
	vec dir;
	vec sup;
	vec rdir;
public:
	Camera(vec org = vec(5), vec lookAt = vec(0), double fov = 50., vec up = vec(0, 1, 0))
		: org(org), lookAt(lookAt), fov(fov), up(up) {
		dir = lookAt - org;
		rdir = normalize(cross(dir, up));
		sup = normalize(cross(rdir, dir));
	}
	Ray camRay(const int & w, const int & h, const vec2 & cp) {
		vec npc = org + normalize(dir);

		double ratio = w * 1. / h;
		double rh = tan(fov / 2.*rad1);
		double rw = rh * ratio;

		double px = fit(cp.x*1. / w, 0, 1, -1, 1);
		double py = fit(cp.y*1. / h, 0, 1, -1, 1);

		return Ray(org, npc + rdir * px*rw + sup * py*rh - org);
	}
	vec DIR() {
		return dir;
	}
	vec SUP() {
		return sup;
	}
	vec RDIR() {
		return rdir;
	}
};

struct BMHead {
	uchar bm[2] = { 0x42,0x4d };
	uchar size[4];
	uchar keep[4];
	uchar sa[4] = { 0x36,0x00,0x00,0x00 };
	uchar hs[4] = { 0x28,0x00,0x00,0x00 };
	uchar sw[4];
	uchar sh[4];
	uchar dl[2] = { 0x01,0x00 };
	uchar cb[2] = { 0x18,0x00 };
	uchar null[24];
	BMHead(int w, int h) {
		for (int i = 0; i < 4; i++)
			keep[i] = 0x00;
		for (int i = 0; i < 24; i++)
			null[i] = 0x00;

		int fs = w * h * 3 + 54;
		for (int i = 0; i < 4; i++) {
			size[i] = (uchar)(fs % 256);
			fs /= 256;
		}
		for (int i = 0; i < 4; i++) {
			sw[i] = (uchar)(w % 256);
			w /= 256;
		}
		for (int i = 0; i < 4; i++) {
			sh[i] = (uchar)(h % 256);
			h /= 256;
		}
	}
	uchar operator[](int i)const {
		return bm[i];
	}
};

void BMofs(ofstream & ofs,const vec_c & c) {
	ofs << /*(uint8_t)((uint)round())*/ c.r
		<< /*(uint8_t)((uint)round())*/ c.g
		<< /*(uint8_t)((uint)round())*/ c.b;
}
void ofsWrite(ofstream & ofs, const vec & c) {
	ofs << (int)round(c.x)
		<< " " << (int)round(c.y)
		<< " " << (int)round(c.z)
		<< endl;
}

istream & operator>>(istream & is, vec & v) {
	cout << "\n vec\n  x:";
	is >> v.x;
	cout << "  y:";
	is >> v.y;
	cout << "  z:";
	is >> v.z;
	return is;
}

vec gamma(const vec & c) {
	return vec(
		sqrt(c.x),
		sqrt(c.y),
		sqrt(c.z)
	);
}
vec atangamma(const vec & c) {
	return vec(
		atan(c.x*pia / 2),
		atan(c.y*pia / 2),
		atan(c.z*pia / 2)
	);
}
vec singamma(const vec & c) {
	return vec(
		sin(c.x*pia / 2),
		sin(c.y*pia / 2),
		sin(c.z*pia / 2)
	);
}

inline void randomAA(vector<vec> & colors, int w, int h, int AA_radius = 1, int iteration_times = 2) {
	for (int i = 0; i < colors.size(); i++) {
		int x = i % w;
		int y = i / w;

		vec color = colors[i];
//#pragma omp parallel for
		for (int j = 0; j < iteration_times; j++) {
			int ox, oy;
			ox = fit(round(drand48()*AA_radius), 0, AA_radius, AA_radius*-1, AA_radius);
			oy = fit(round(drand48()*AA_radius), 0, AA_radius, AA_radius*-1, AA_radius);
			if (x < AA_radius)
				ox = round(drand48()*AA_radius);
			if (x > w - AA_radius)
				ox = fit(round(drand48()*AA_radius), 0, AA_radius, AA_radius*-1, 0);
			if (y < AA_radius)
				oy = round(drand48()*AA_radius);
			if (y > h - AA_radius)
				oy = fit(round(drand48()*AA_radius), 0, AA_radius, AA_radius*-1, 0);
			color += colors[i + ox + oy * w];
		}
		color /= (iteration_times + 1);

		colors[i] = color;
	}
}

Sphere skydome(vec(0), 100, new diffuse);
Sphere splt(vec(-.8, 5.2, .9), .5, new diffuse);
vec shading(obj * scenes, const Ray & ray, int iteration_times = 8, const int & useLight = 1, const double & ptllum = 50) {
	if (iteration_times > 50)
		iteration_times = 50;
	hitInfo hi = scenes->hit(ray);
	if (hi.isHit) {
		if (iteration_times > 0) {
			scatterInfo si = hi.ptmat->scatter(hi, ray);
			return shading(scenes, si.outRay, iteration_times - 1, useLight, ptllum) * si.mult;
		}
		else return vec(0);
	}
	else {
		if (useLight) {
			hitInfo lihi = splt.hit(ray);
			if (lihi.isHit) {
				return vec(.99, .968, 0.957) * ptllum;
			}
			else {
				hitInfo hi = skydome.hit(ray);
				double lum = fit(hi.hitPos.y, -100, 100, .5, 1);
				return vec(1)*(1 - lum) + vec(.48, .68, .9)*lum;
			}
		}
		else {
			hitInfo hi = skydome.hit(ray);
			double lum = fit(hi.hitPos.y, -100, 100, .5, 1);
			return vec(1)*(1 - lum) + vec(.48, .68, .9)*lum;
		}
	}
}

int main()
{
	int w = 1024;
	int h = 768;

	int sample_times = 16;
	int useptLight = 0;
	double ptllum = 50;
	int depth = 16;
	int RAAr = 1;
	int RAAitt = 2;
	int sw = 0; //randomAA
	int othersw = 0;
	vec campos = vec(.15, 2.3, 4.5)/*vec(-1.1,3.6,-3.8)*/;
	vec camlookpos = vec(.11, .8, 0);
	double camfov = 66;

	double snell_01_IOR = 3.1;
	double snell_02_IOR = 1.33;

	int floorM = 0;
	double floorMrough = .1;
	int othsw02 = 0;

	char gammasw = 'b';

	
	{
		cout << "\n\tDimension Faker 光线投射/追踪 模拟 LT";
		cout << "    bilibili视频特供版";
		cout << "\n\n\t\t\tcode by Ark_Noya\n\t\t\tArk_Noya@qq.com\n\n\n\n";

		cout << " ( 以下参数不想细看可以全填成 0 )\n ( 参数会自动设置成默认值 )\n\n";
		cout << " 输入画面高宽度：\n\twidth : ";
		cin >> w;
		if (w < 10) w = 1024;
		cout << "\theigh : ";
		cin >> h;
		if (h < 10) h = 768;
		
		cout << "\n 是否使用模拟点灯光 : ( 1 -> yes  /  0 -> no ) (使用后采样数会自动调成 128 (默认8) ) : ";
		cin >> useptLight;
		if (useptLight) {
			sample_times = 128;
			cout << "  灯光亮度 : (建议在 50 以上) : ";
			cin >> ptllum;
			if (ptllum < 1)
				ptllum = 50;
		}
		
		cout << "\n 光线深度 : ";
		cin >> depth;
		if (depth < 1) depth = 8;

		if (useptLight) {
			cout << " 采样数 : ";
			cin >> sample_times;
			if (sample_times < 64) sample_times = 128;
		}
		else {
			cout << " 采样数 : ";
			cin >> sample_times;
			if (sample_times < 1) sample_times = 4;
		}
		
		cout << "\n\n 是否使用randomAA  (1 -> yes  /  0 -> no) : ";
		cin >> sw;
		if (sw) {
			cout << "\n randomAA半径 : ";
			cin >> RAAr;
			if (RAAr < 1) RAAr = 1;
			cout << "\n randomAA采样次数 : ";
			cin >> RAAitt;
			if (RAAitt < 1) RAAitt = 2;
		}
		
		cout << "\n\n gamma颜色管理方案 :    ( 0.无  a.gamma  b.atangamma  c.singamma ) : ";
		cin >> gammasw;

		cout << "\n\n 显示其他参数 (1 -> yes  /  0 -> no) : ";
		cin >> othsw02;
		if (othsw02) {
			cout << "\n  地板材质 :  (1 -> 金属  /  0 -> 棋盘) : ";
			cin >> floorM;
			if (floorM) {
				cout << "\n  地板金属材质 Roughness : (建议在 .09 以下) : ";
				cin >> floorMrough;
			}
			cout << "\n\n 显示更多 (1 -> yes  /  0 -> no) : ";
			cin >> othersw;
			if (othersw) {
				cout << "\n  相机位置 :  (默认位置 : (.15, 2.3, 4.5)  一个推荐位置 : (-1.1, 3.6, -3.8) ) : ";
				cin >> campos;
				int otsw = 0;
				cout << "\n\n 显示更多 (1 -> yes  /  0 -> no) : ";
				cin >> otsw;
				if (otsw) {
					cout << "\n  摄像机视点 : (默认位置 : (.11, .8, 0) ) : ";
					cin >> camlookpos;
					cout << "\n  摄像机视野 : (默认 fov : 66 ) : ";
					cin >> camfov;
					int otsw2 = 0;
					cout << "\n\n 显示更多 (1 -> yes  /  0 -> no) : ";
					cin >> otsw2;
					if (otsw2) {
						cout << "\n  snell球_01 IOR : (默认为 3.1 ) : ";
						cin >> snell_01_IOR;
						if (snell_01_IOR < 1)
							snell_01_IOR = 3.1;
						cout << "\n  snell球_02 IOR : (默认为 1.33 ) : ";
						cin >> snell_02_IOR;
						if (snell_01_IOR < 1)
							snell_01_IOR = 1.33;
					}
				}
			}
		}

		cout << "\n\n\n\n\n   信息统计 :\n 画面高宽 :\n  width : " << w << "\n  heigh : " << h << "\n\n";
		cout << " 开启模拟点灯光 : " << (useptLight == 1 ? "NO" : "OFF");
		cout << "\n\n 光线深度 : " << depth << "\n 采样数 : " << sample_times << "\n\n randomAA开关 : ";
		cout << (sw == 1 ? "ON" : "OFF");
		if (sw) {
			cout << "\n  randomAA半径 : " << RAAr;
			cout << "\n  randomAA采样次数 : " << RAAitt;
		}
		cout << "\n\n 摄像机位置 : " << campos;
		cout << "\n 摄像机视点 : " << camlookpos;
		cout << "\n 摄像机视野 : " << camfov;
		cout << "\n\n\n\n\n";
	}
	
	//adv
	int advon = 0;
	vec advvec = vec(.11, .8, 0) - vec(-1.1, 3.6, -3.8);
	if (campos.x<-.9 && campos.x>-1.3 && campos.y<3.8 && campos.y>3.4 && campos.z<-3.6 && campos.z>-4.) {
		if (useptLight)
			if (ptllum > 49)
				if (sample_times > 63)
					if (floorM == 0)
						if (angle(advvec, camlookpos - campos) < 15 * rad1)
							advon = 1;
	}

	//out file stream
	ofstream ppmofs("result.ppm");
	ppmofs << "P3\n" << w << " " << h << "\n255\n";

	//obj
	Camera cam0(campos, camlookpos, camfov);
	material * gridDIF = new gridDif(vec(.8, .7, .89));
	material * gridMET = new metal(vec(.8, .7, .89), floorMrough);
	material * griMat;
	if (floorM == 1)
		griMat = gridMET;
	else griMat = gridDIF;
	obj * list[8];
	list[0] = new Sphere(vec(0.05, 1.11, 0), 1.1, new toonDif(vec(.4, .6, .8)));
	list[1] = new Sphere(vec(0,-200.01,0), 200, griMat);
	list[2] = new Sphere(vec(1.01 + 2.4, 2.51, 0), 2.5, new metal(vec(.9, .6, .5),.09));
	list[3] = new Sphere(vec(-1.01 - 1.9, 1.91, 0), 1.9, new metal(vec(.92, .75, .6), .028));
	list[4] = new Sphere(vec(2.1, 1.501, 4.5), 1.2, new diffuse(vec(.5, .95, .6)));
	list[5] = new Sphere(vec(-.4, .71, .7 + 1.1), .7, new snell(vec(.89, .91, .95), snell_01_IOR, .015));
	list[6] = new Sphere(vec(.45, .81, -.81 - .68), .8, new snell(vec(.98), snell_02_IOR));
	//list[6] = new Sphere(vec(48.9,0,0), 50, new snell(vec(.98), snell_02_IOR));
	list[7] = new Sphere(vec(-1.8, 1.301, 3.8), 1.1, new cosGridDif(vec(.95, .52, .5),vec(0)));
	obj * scenes = new obj_list(list, 8);

	//reg
	vector<vec> colors;
	vec zv(0);
	int t0 = time(NULL);
	for (int i = 0; i < w*h; i++)
		colors.push_back(zv);
	int t1 = time(NULL);
	cout << "reg OK ! time used : " << t1 - t0 << " second !" << endl;
	
	//rendering
#pragma omp parallel for
	for (int i = 0; i < colors.size(); i++) {
		int x = i % w;
		int y = h - i / w;
		vec2 cp(x, y);
		Ray cr = cam0.camRay(w, h, cp);

		vec c(0);
		for(int j=0;j<sample_times;j++)
			c += shading(scenes, cr, depth, useptLight, ptllum);
		c /= sample_times;

		c = fit01(c, 0, 1);

		switch (gammasw)
		{
		case '0':
			break;

			case 'a':
			case 'A':
			case '1':
				c = gamma(c);
				break;

			case 'b':
			case 'B':
			case '2':
				c = atangamma(c);
				break;

			case 'c':
			case 'C':
			case '3':
				c = singamma(c);
				break;

		default:
			break;
		}

		colors[i] = c * 254;
	}
	int t2 = time(NULL);
	cout << "Render OK ! time used : " << t2 - t1 << " second !" << endl;

	//AA
	if (sw) randomAA(colors, w, h, RAAr, RAAitt);
	int t3 = time(NULL);
	cout << "randomAA OK ! time used : " << t3 - t2 << " second !" << endl;

	//Write file
	cout << "\n Writing PPM image file ...";
	for (int i = 0; i < colors.size(); i++)
		ofsWrite(ppmofs, colors[i]);
	
	int t4 = time(NULL);
	cout << "\nWrite File OK ! time used : " << t4 - t3 << " second !\n" << endl;

	ppmofs.close();
	system("result.ppm");
	if (advon) {
		ofstream cmdofs("adv.cmd");
		cmdofs << "echo off\ntitle Dimension Faker 隐藏成就\ncolor B\ncls\necho ####################\necho    恭喜达成成就：\necho      找到你了！\necho ####################\npause";
		system("start adv.cmd");
	}
	system("pause");
	return 0;
}