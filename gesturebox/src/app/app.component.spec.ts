import { By } from '@angular/platform-browser';
import { DebugElement, CUSTOM_ELEMENTS_SCHEMA, EventEmitter } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { RouterTestingModule } from '@angular/router/testing';
import { TestBed, async, ComponentFixture } from '@angular/core/testing';

import { AppLocationsService } from '@kognifai/poseidon-ng-applocationsservice';
import { AuthenticationService as AuthenticationSvc } from '@kognifai/poseidon-authenticationservice';
import { AuthenticationService } from '@kognifai/poseidon-ng-authenticationservice';
import { ConfigurationService } from '@kognifai/poseidon-ng-configurationservice';
import { CookieService } from '@kognifai/poseidon-cookieservice';
import { FooterModule } from '@kognifai/poseidon-ng-footer-component';
import { HeaderModule } from '@kognifai/poseidon-header-component';
import { IConfiguration } from '@kognifai/poseidon-configurationinterface';
import { InitializeService } from '@kognifai/poseidon-ng-initialize-service';
import { MessageModule } from '@kognifai/poseidon-ng-message-component';
import { MessageService } from '@kognifai/poseidon-message-service';
import { NavigationService, INavigationService, NavigationItem, INavigationItem } from '@kognifai/poseidon-ng-navigationservice';
import { SidebarsVisibilityService } from '@kognifai/poseidon-sidebar-visibilityservice';
import { ToolsMenuService } from '@kognifai/poseidon-ng-toolsmenuservice';

import { AppComponent } from './app.component';
import { MainComponent } from './main/main.component';

describe('AppComponent', () => {
  let component: AppComponent;
  let fixture: ComponentFixture<AppComponent>;
  let de: DebugElement;

  let navigationServiceStub: NavigationService;
  let authenticationSvcStub: AuthenticationSvc;
  let authenticationServiceStub: AuthenticationService;
  let cookieServiceStub: CookieService;
  let messageServiceStub: MessageService;
  let configurationServiceStub: ConfigurationService<IConfiguration>;
  let appLocationsServiceStub: AppLocationsService;
  let initializeServiceStub: InitializeService;
  let toolsMenuServiceStub: ToolsMenuService;
  let sidebarsVisibilityServiceStub: SidebarsVisibilityService;

  beforeEach(async(() => {
    navigationServiceStub = new NavigationService(null, null);
    authenticationSvcStub = new AuthenticationSvc();
    authenticationServiceStub = new AuthenticationService(null);
    cookieServiceStub = new CookieService();
    messageServiceStub = new MessageService();
    configurationServiceStub = new ConfigurationService(null);
    (<any>configurationServiceStub).configuration = {};
    configurationServiceStub.load = () => Promise.resolve();
    appLocationsServiceStub = <AppLocationsService> { get: null };
    spyOn(appLocationsServiceStub, 'get').and.callFake(() => Promise.resolve());
    initializeServiceStub = new InitializeService(
      cookieServiceStub, authenticationServiceStub, configurationServiceStub, navigationServiceStub, appLocationsServiceStub);
    initializeServiceStub.initialize = () => new Observable<void>(observer => observer.complete());
    toolsMenuServiceStub = {
      items: [],
      register: (items) => { },
      clear: () => { }
    };
    sidebarsVisibilityServiceStub = {
      pageOverlayActive: false,
      pageOverlayAnimating: false,
      navigationPanelHidden: false,
      navigationPanelActive: false,
      navigationButtonActive: false,
      toolsPanelHidden: false,
      toolsPanelActive: false,
      toolsButtonActive: false
    };

    TestBed.configureTestingModule({
      imports: [RouterTestingModule, HttpClientModule, FormsModule],
      declarations: [AppComponent],
      providers: [
        { provide: MessageService, useValue: messageServiceStub },
        { provide: NavigationService, useValue: navigationServiceStub },
        { provide: AuthenticationSvc, useValue: authenticationSvcStub },
        { provide: AuthenticationService, useValue: authenticationServiceStub },
        { provide: InitializeService, useValue: initializeServiceStub },
        { provide: ConfigurationService, useValue: configurationServiceStub },
        { provide: ToolsMenuService, useValue: toolsMenuServiceStub },
        { provide: SidebarsVisibilityService, useValue: sidebarsVisibilityServiceStub }
      ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA]
    }).compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(AppComponent);
    component = fixture.componentInstance;
    de = fixture.debugElement;
    fixture.detectChanges();
  });

  it('should create the app', () => {
    expect(component).toBeTruthy();
  });

  it('should create tools menu', () => {
    const toolsMenuElement = de.query(By.css('app-tools-menu'));
    expect(toolsMenuElement).toBeTruthy();
  });

  it('tools menu should be hidden initially', () => {
    const toolsMenuElement = de.query(By.css('app-tools-menu'));
    expect(toolsMenuElement.attributes['class']).toContain('kx-is-hidden');
  });
});
